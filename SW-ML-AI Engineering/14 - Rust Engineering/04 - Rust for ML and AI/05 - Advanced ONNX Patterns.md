# 🔬 Advanced ONNX Patterns

## 🎯 Learning Objectives
- Master ONNX Runtime's graph optimization levels and understand what each transformation does to your model.
- Implement custom ONNX operators in Rust and integrate them with ONNX Runtime sessions.
- Build a complete INT8 quantization pipeline: calibration → quantization → deployment.
- Configure intra-op and inter-op parallelism for maximum throughput on multi-core CPUs and multi-GPU setups.
- Integrate ONNX Runtime with Tokio async runtime for production Rust servers.

## Introduction

Basic ONNX Runtime usage — loading a model, running inference, extracting outputs — is straightforward. But production ML systems require deeper understanding: graph optimization levels that can silently break your model, custom operators for unsupported architectures, quantization pipelines that balance accuracy and speed, and thread-level parallelism configuration that can make or break your P99 latency. This note covers the advanced patterns that separate prototype code from production-grade ONNX Runtime deployments in Rust.

These patterns build directly on [[03 - ONNX Runtime Rust|the core ONNX Runtime concepts]] and complement [[04 - ONNX vs Candle: Choosing the Right Runtime|the runtime decision framework]]. They are essential when you've chosen ONNX Runtime for its GPU performance and need to squeeze every millisecond of latency from your inference pipeline.

## 1. 🧠 Graph Optimization Levels — Theory and Practice

### What Each Optimization Level Does

ONNX Runtime applies a sequence of graph transformations that rewrite the computational graph for efficiency. The levels control how aggressive these transformations are:

| Level | Name | Transformations | Risk |
|-------|------|----------------|------|
| **Level 0** | DisableAll | None | Safe — no graph changes |
| **Level 1** | Basic | Constant folding, redundant node elimination, redundant squeeze/unsqueeze removal | Very safe |
| **Level 2** | Extended | Level 1 + node fusion (Conv+BN, Conv+Relu, Gemm+Relu), partial constant folding | Safe for standard models |
| **Level 3** | Layout | Level 2 + layout optimization (NCHW→NHWC for CPU EP), extended fusion (Conv+BN+Relu, Conv+Clip) | Minor risk with custom shapes |
| **Level 99** | All | Level 3 + all experimental optimizations, aggressive fusion, multi-node pattern matching | Can break models with dynamic shapes |

### Mathematical Basis of Constant Folding

Constant folding evaluates subgraphs where all inputs are known at graph-load time:

```
If input(v) = {c₁, c₂, ..., cₙ} where all cᵢ are constants:
    Replace v with op(v).evaluate()  // compute at load time
    Remove v from runtime graph
```

Example: `y = matmul(X, W) + b` where `W` and `b` are initializers but `X` is a runtime input cannot be folded. But `z = matmul(W₁, W₂)` where both are known can be pre-computed.

### Node Fusion Patterns

```
┌────────────────────────────────────────────────────────────┐
│                    Layer Fusion Examples                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Before Fusion:              After Fusion:                  │
│  ┌─────────┐                ┌──────────────┐               │
│  │  Conv2D │                │              │               │
│  └────┬────┘                │  ConvBNRelu  │               │
│       │                     │  (single      │               │
│  ┌────▼────┐                │   fused       │               │
│  │BatchNorm│                │   kernel)     │               │
│  └────┬────┘                │              │               │
│       │                     └──────────────┘               │
│  ┌────▼────┐                                                │
│  │  ReLU   │       Reduces: 3 kernel launches → 1           │
│  └─────────┘       Eliminates: 2 intermediate tensors       │
│                                                             │
│  Fused ops: Conv+BN, Conv+ReLU, Conv+BN+ReLU,               │
│             Conv+Clip, Gemm+ReLU, Conv+Add+ReLU,            │
│             MatMul+Add, MatMul+Add+ReLU                     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Extended Fusion for Transformer Models

Transformer architectures benefit from attention-specific fusions:

```
MultiHeadAttention fusion pattern:
  Original:  MatMul(Q,K) → Scale → Softmax → MatMul(V) → MatMul(proj)
  Optimized: FusedAttention (single GPU kernel, flash attention pattern)

LayerNorm fusion:
  Original:  ReduceMean → Sub → Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul → Add
  Optimized: LayerNorm (single kernel, skip intermediate tensors)
```

### Configuring Graph Optimizations in Rust

```rust
use ort::{GraphOptimizationLevel, Session};

fn create_optimized_session(model_path: &str) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(8)?
        .with_inter_op_threads(4)?
        .with_graph_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)
}

fn create_safe_session(model_path: &str) -> ort::Result<Session> {
    // Use Level 1 for production when debugging accuracy issues
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .commit_from_file(model_path)
}
```

## 2. 📐 Custom Operators — Extending ONNX Runtime in Rust

### Why Custom Operators Exist

The ONNX standard defines 180+ operators, but real-world models use:
- Novel attention mechanisms not yet standardized
- Domain-specific operations (e.g., physics simulation, signal processing)
- Proprietary layers that can't be expressed as ONNX standard ops
- Custom quantization/dequantization schemes

### Architecture of a Custom Operator

```
┌──────────────────────────────────────────────────────────────────┐
│                Custom Operator Registration Flow                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐                                            │
│  │ Define Op Schema  │  .onnx domain + inputs/outputs/types       │
│  └────────┬─────────┘                                            │
│           │                                                       │
│  ┌────────▼─────────┐                                            │
│  │ Register with ORT │  ort::CustomOpDomain + kernel factory      │
│  └────────┬─────────┘                                            │
│           │                                                       │
│  ┌────────▼─────────┐                                            │
│  │  Implement Kernel │  Compute() method: input tensors → output  │
│  └────────┬─────────┘                                            │
│           │                                                       │
│  ┌────────▼─────────┐                                            │
│  │  Load Model       │  Session reads custom domain from .onnx   │
│  └────────┬─────────┘                                            │
│           │                                                       │
│  ┌────────▼─────────┐                                            │
│  │  Run Inference    │  ORT dispatches to your kernel at runtime  │
│  └──────────────────┘                                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Custom Operator Implementation in Rust

```rust
use ort::{CustomOp, CustomOpDomain, Kernel, Session, Tensor};
use std::sync::Arc;

#[derive(Debug)]
struct SiluOp;

impl CustomOp for SiluOp {
    fn name(&self) -> &str {
        "Silu"
    }

    fn domain(&self) -> &str {
        "custom.ai"
    }

    fn version(&self) -> i32 {
        1
    }

    fn create_kernel(
        &self,
        _inputs: &[ort::custom::TensorType],
        _outputs: &[ort::custom::TensorType],
    ) -> ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(SiluKernel))
    }
}

struct SiluKernel;

impl Kernel for SiluKernel {
    fn compute(&self, context: &mut ort::custom::ComputeContext) -> ort::Result<()> {
        let input = context.get_input(0)?;
        let input_data = input.try_extract_tensor::<f32>()?;
        let (_shape, data) = input_data;

        let output_data: Vec<f32> = data.iter()
            .map(|&x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            })
            .collect();

        let output_shape = input.shape().to_vec();
        let output = Tensor::from_array((output_shape, output_data))?;
        context.set_output(0, &output)?;

        Ok(())
    }
}

fn register_and_run() -> ort::Result<()> {
    let mut domain = CustomOpDomain::new("custom.ai");
    domain.add(SiluOp);

    ort::init()
        .with_name("custom_ops_example")
        .with_custom_op_domain(domain)
        .commit()?;

    let session = Session::builder()?
        .commit_from_file("model_with_silu.onnx")?;

    Ok(())
}
```

### Custom Operator with CUDA Acceleration

```rust
use ort::custom::{CustomOp, Kernel, TensorType, ComputeContext};

struct CudaSiluKernel;

impl Kernel for CudaSiluKernel {
    fn compute(&self, context: &mut ComputeContext) -> ort::Result<()> {
        let input = context.get_input(0)?;

        // Check if input is on GPU
        if context.is_gpu_input(0) {
            let gpu_ptr = context.get_gpu_input_ptr(0)?;
            let output_ptr = context.get_gpu_output_ptr(0)?;
            let element_count: usize = context.input_shape(0)?
                .iter()
                .product();

            unsafe {
                launch_silu_cuda_kernel(
                    gpu_ptr as *const f32,
                    output_ptr as *mut f32,
                    element_count,
                )?;
            }
        } else {
            // CPU fallback
            let input_data = input.try_extract_tensor::<f32>()?;
            let (_shape, data) = input_data;
            let output_data: Vec<f32> = data.iter()
                .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
                .collect();
            let output_shape = input.shape().to_vec();
            let output = Tensor::from_array((output_shape, output_data))?;
            context.set_output(0, &output)?;
        }

        Ok(())
    }
}

unsafe fn launch_silu_cuda_kernel(
    input: *const f32,
    output: *mut f32,
    count: usize,
) -> ort::Result<()> {
    // In practice: use cust/cudarc crate to launch a CUDA kernel
    // that computes SiLU element-wise in parallel on the GPU
    Ok(())
}
```

## 3. 💻 INT8 Quantization Pipeline

### Dynamic vs Static Quantization

| Aspect | Dynamic Quantization | Static Quantization |
|--------|---------------------|---------------------|
| **When calib happens** | During inference | Before deployment |
| **Scale/zero-point** | Per-batch, runtime | Pre-computed from calibration |
| **Accuracy** | Slightly lower | Higher (calibration matches distribution) |
| **Speed** | Slower (recomputes each batch) | Faster (pre-computed) |
| **Calibration data needed** | No | Yes (100-1000 representative samples) |
| **Best for** | Unknown input distribution | Fixed, known input distribution |

### INT8 Quantization Math

For a tensor $W$ with range $[W_{min}, W_{max}]$:

```
scale = (W_max - W_min) / (2^8 - 1)    // for unsigned INT8
zero_point = round(-W_min / scale)
quantized_value = clamp(round(W_i / scale) + zero_point, 0, 255)
```

Per-channel quantization (better accuracy for convolutions):

```
For channel c:
    scale_c = (W_max_c - W_min_c) / 255
    zero_point_c = round(-W_min_c / scale_c)
    W_q[c, i, j] = clip(round(W[c, i, j] / scale_c) + zero_point_c, 0, 255)
```

### Complete Quantization Pipeline

```rust
use ort::{
    GraphOptimizationLevel, Session, SessionBuilder, Tensor,
    quantization::{QuantFormat, QuantType},
};
use std::path::{Path, PathBuf};
use std::fs;

struct QuantizationPipeline {
    input_path: PathBuf,
    output_dir: PathBuf,
    calibration_samples: usize,
}

impl QuantizationPipeline {
    fn new(input_path: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            input_path,
            output_dir,
            calibration_samples: 256,
        }
    }

    fn run(&self) -> ort::Result<PipelineResult> {
        let original_size = fs::metadata(&self.input_path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Step 1: Generate or load calibration data
        let calibration_data = self.load_calibration_data()?;

        // Step 2: Apply graph optimization first
        let optimized_path = self.output_dir.join("stage1_optimized.onnx");
        self.optimize_graph(&optimized_path)?;

        // Step 3: Perform static INT8 quantization
        let quantized_path = self.output_dir.join("stage2_int8.onnx");
        self.quantize_static_int8(
            &optimized_path,
            &quantized_path,
            &calibration_data,
        )?;

        // Step 4: Validate quantized model
        let quantized_size = fs::metadata(&quantized_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let accuracy = self.validate_accuracy(&quantized_path, &calibration_data)?;

        Ok(PipelineResult {
            original_size_bytes: original_size,
            quantized_size_bytes: quantized_size,
            size_reduction_ratio: original_size as f64 / quantized_size as f64,
            accuracy_drop_percent: accuracy,
        })
    }

    fn load_calibration_data(&self) -> ort::Result<Vec<Vec<f32>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let data: Vec<Vec<f32>> = (0..self.calibration_samples)
            .map(|_| {
                (0..512)
                    .map(|_| rng.gen_range(-2.0..2.0))
                    .collect()
            })
            .collect();

        Ok(data)
    }

    fn optimize_graph(&self, output_path: &Path) -> ort::Result<()> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(&self.input_path)?;

        session.save_optimized_model(output_path)?;

        println!("Stage 1: Graph optimization completed");
        Ok(())
    }

    fn quantize_static_int8(
        &self,
        input_path: &Path,
        output_path: &Path,
        calibration_data: &[Vec<f32>],
    ) -> ort::Result<()> {
        // Convert calibration data to tensors
        let mut dataset = Vec::new();
        for sample in calibration_data {
            let tensor = Tensor::from_array(
                ([1, sample.len()], sample.clone())
            )?;
            dataset.push(tensor);
        }

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_static_quantization(
                QuantFormat::QDQ,
                QuantType::Int8,
                &dataset,
            )?
            .with_intra_threads(8)?
            .commit_from_file(input_path)?;

        session.save_optimized_model(output_path)?;

        println!(
            "Stage 2: Static INT8 quantization completed ({} calibration samples)",
            calibration_data.len()
        );
        Ok(())
    }

    fn validate_accuracy(
        &self,
        model_path: &Path,
        validation_data: &[Vec<f32>],
    ) -> ort::Result<f64> {
        // Compare quantized model outputs to reference
        let reference_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::DisableAll)?
            .commit_from_file(&self.input_path)?;

        let quantized_session = Session::builder()?
            .commit_from_file(model_path)?;

        let mut total_diff = 0f64;
        let mut count = 0;

        for sample in validation_data.iter().take(50) {
            let input = Tensor::from_array(
                ([1, sample.len()], sample.clone())
            )?;

            let ref_output = reference_session
                .run(ort::inputs!["input" => input.clone()])?;
            let quant_output = quantized_session
                .run(ort::inputs!["input" => input])?;

            let ref_data = ref_output[0]
                .try_extract_tensor::<f32>()?
                .1.to_vec();
            let quant_data = quant_output[0]
                .try_extract_tensor::<f32>()?
                .1.to_vec();

            for (r, q) in ref_data.iter().zip(quant_data.iter()) {
                total_diff += (r - q).abs() as f64;
                count += 1;
            }
        }

        let avg_diff = total_diff / count as f64;
        Ok(avg_diff)
    }
}

struct PipelineResult {
    original_size_bytes: u64,
    quantized_size_bytes: u64,
    size_reduction_ratio: f64,
    accuracy_drop_percent: f64,
}
```

## 4. 📊 Multi-Threading and Memory Management

### Intra-Operatoryll vs Inter-Op Parallelism

```
┌──────────────────────────────────────────────────────────────────┐
│              Intra-Op vs Inter-Op Parallelism                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Intra-Op Parallelism (with_intra_threads)                         │
│  ─────────────────────────────────────                             │
│  Parallelism within a single operator. Example: MatMul             │
│                                                                   │
│  A₁₁ A₁₂ A₁₃     B₁₁ B₁₂     Thread 1: A₁₁×B₁₁ + A₁₂×B₂₁       │
│  A₂₁ A₂₂ A₂₃  ×  B₂₁ B₂₂     Thread 2: A₁₁×B₁₂ + A₁₂×B₂₂       │
│                   B₃₁ B₃₂     Thread 3: A₂₁×B₁₁ + A₂₂×B₂₁       │
│                                Thread 4: A₂₁×B₁₂ + A₂₂×B₂₂       │
│                                                                   │
│  Rule: intra_threads ≈ min(num_cores, rows × inner_dim / 1024)  │
│                                                                   │
│  Inter-Op Parallelism (with_inter_op_threads)                       │
│  ─────────────────────────────────────                             │
│  Parallelism across independent subgraphs. Example:                │
│                                                                   │
│         ┌─────────┐                                               │
│    ┌────┤ Branch A├────┐                                          │
│    │    └─────────┘    │                                          │
│  ┌─┴──┐              ┌─┴───┐     Branch A runs on Thread 1        │
│  │Split│             │Merge│     Branch B runs on Thread 2        │
│  └─┬──┘              └─┬───┘     Both execute concurrently        │
│    │    ┌─────────┐    │                                          │
│    └────┤ Branch B├────┘                                          │
│         └─────────┘                                               │
│                                                                   │
│  Rule: inter_threads ≤ number of independent parallel branches   │
│                                                                   │
│  Total thread pool: intra_threads + inter_threads ≤ total_cores  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Thread Affinity and NUMA-Aware Configuration

```rust
use ort::{GraphOptimizationLevel, Session, SessionBuilder};
use num_cpus;

fn create_numa_optimized_session(model_path: &str) -> ort::Result<Session> {
    let physical_cores = num_cpus::get_physical();
    let numa_nodes = detect_numa_nodes(); // Platform-specific

    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(physical_cores / numa_nodes)?
        .with_inter_op_threads(numa_nodes)?
        .with_intra_op_num_threads(physical_cores / numa_nodes)?
        .commit_from_file(model_path)
}

// In production, map session to specific NUMA node and pin threads:
fn create_session_for_numa_node(
    model_path: &str,
    numa_node: usize,
    cores: Vec<usize>,
) -> ort::Result<Session> {
    // Bind session threads to specific CPU cores
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(cores.len())?
        .commit_from_file(model_path)
}

fn detect_numa_nodes() -> usize {
    // On Linux: count directories in /sys/devices/system/node/node*
    // Returns 1 on non-NUMA systems
    std::path::Path::new("/sys/devices/system/node/node1")
        .exists()
        .then_some(2)
        .unwrap_or(1)
}
```

### Memory Management: Arena Allocators and IO Binding

```rust
use ort::{Session, Tensor, MemoryInfo, AllocatorType, MemType};

fn io_bound_inference(session: &Session, input_data: &[f32]) -> ort::Result<Vec<f32>> {
    // Create memory info for pinned (page-locked) memory
    // Pinned memory enables faster CPU↔GPU transfers via DMA
    let memory_info = MemoryInfo::new(
        "Cpu",
        AllocatorType::Arena,
        0,
        MemType::CpuInput,
    )?;

    // Create tensor with IO binding (zero-copy where possible)
    let input_shape = vec![1, input_data.len() as i64];
    let input_tensor = Tensor::from_array(
        memory_info,
        input_data.to_vec(),
        &input_shape,
    )?;

    let outputs = session.run(ort::inputs!["input" => input_tensor])?;

    let output = outputs[0].try_extract_tensor::<f32>()?;
    let (_, data) = output;
    Ok(data.to_vec())
}
```

### Session Pooling for Concurrent Inference

```rust
use ort::Session;
use std::sync::Arc;
use tokio::sync::{Semaphore, RwLock};
use std::collections::VecDeque;

struct SessionPool {
    sessions: RwLock<VecDeque<Arc<Session>>>,
    semaphore: Semaphore,
    model_path: String,
}

impl SessionPool {
    fn new(model_path: &str, pool_size: usize) -> ort::Result<Self> {
        let mut sessions = VecDeque::with_capacity(pool_size);

        for _ in 0..pool_size {
            let session = Session::builder()?
                .with_optimization_level(
                    ort::GraphOptimizationLevel::Level3
                )?
                .commit_from_file(model_path)?;
            sessions.push_back(Arc::new(session));
        }

        Ok(Self {
            sessions: RwLock::new(sessions),
            semaphore: Semaphore::new(pool_size),
            model_path: model_path.to_string(),
        })
    }

    async fn acquire(&self) -> Arc<Session> {
        let permit = self.semaphore.acquire().await.unwrap();
        let session = {
            let mut sessions = self.sessions.write().await;
            sessions.pop_front().unwrap()
        };

        // Store the permit in the session wrapper to auto-release
        // In practice, use a RAII guard that returns the session to the pool
        std::mem::forget(permit);
        session
    }

    async fn release(&self, session: Arc<Session>) {
        let mut sessions = self.sessions.write().await;
        sessions.push_back(session);
        self.semaphore.add_permits(1);
    }
}
```

## 5. 🔄 Tokio Async Integration

### Async Inference Wrapper

```rust
use ort::Session;
use std::sync::Arc;
use tokio::task;

struct AsyncInferenceEngine {
    session: Arc<Session>,
    runtime: tokio::runtime::Handle,
}

impl AsyncInferenceEngine {
    fn new(model_path: &str) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(
                ort::GraphOptimizationLevel::Level3,
            )?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: Arc::new(session),
            runtime: tokio::runtime::Handle::current(),
        })
    }

    async fn infer_async(
        &self,
        input: Vec<f32>,
        shape: Vec<usize>,
    ) -> ort::Result<Vec<f32>> {
        let session = self.session.clone();

        // Run inference on a blocking thread to avoid
        // starving the tokio async runtime
        task::spawn_blocking(move || {
            let input_tensor = ort::Tensor::from_array(
                (shape, input),
            )?;

            let outputs = session.run(ort::inputs![
                "input" => input_tensor
            ])?;

            let output = outputs[0].try_extract_tensor::<f32>()?;
            let (_, data) = output;
            Ok(data.to_vec())
        })
        .await
        .unwrap()
    }

    async fn infer_batch_async(
        &self,
        batch: Vec<(Vec<f32>, Vec<usize>)>,
    ) -> ort::Result<Vec<Vec<f32>>> {
        let session = self.session.clone();

        task::spawn_blocking(move || {
            batch.into_iter()
                .map(|(data, shape)| {
                    let input_tensor = ort::Tensor::from_array(
                        (shape, data),
                    )?;
                    let outputs = session.run(ort::inputs![
                        "input" => input_tensor
                    ])?;
                    let output = outputs[0].try_extract_tensor::<f32>()?;
                    Ok(output.1.to_vec())
                })
                .collect()
        })
        .await
        .unwrap()
    }
}

impl AsyncInferenceEngine {
    fn schedule_prewarm(&self) {
        let session = self.session.clone();
        tokio::spawn(async move {
            // Run a few dummy inferences to warm up GPU,
            // compile CUDA kernels, and fill caches
            let warmup_data = vec![0.0f32; 512];
            for _ in 0..5 {
                let input = ort::Tensor::from_array(
                    ([1, 512], warmup_data.clone()),
                ).unwrap();
                let _ = session.run(ort::inputs!["input" => input]);
            }
        });
    }
}
```

## 6. 🌍 Model Conversion Pipeline

### PyTorch → ONNX → ORT Optimized → Deployment

```
┌──────────────────────────────────────────────────────────────────────┐
│                    End-to-End Conversion Pipeline                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────┐                                              │
│  │  PyTorch Model      │    model = torch.load("model.pt")            │
│  │  (training)         │                                              │
│  └────────┬───────────┘                                              │
│           │ torch.onnx.export(model, dummy_input, "model.onnx",     │
│           │   opset_version=17, dynamic_axes={'input': {0: 'batch'}})│
│  ┌────────▼───────────┐                                              │
│  │  ONNX Model         │    Standard IR, framework-agnostic          │
│  │  (portable)         │                                              │
│  └────────┬───────────┘                                              │
│           │ onnx.checker.check_model("model.onnx")                   │
│  ┌────────▼───────────┐                                              │
│  │  Validation         │    Verify all ops are standard, shapes ok   │
│  └────────┬───────────┘                                              │
│           │ ort::Session::builder()?.with_optimization_level(3)?.   │
│           │   commit_from_file("model.onnx")?                        │
│  ┌────────▼───────────┐                                              │
│  │  Graph Optimization │    Constant folding, fusion, layout opt     │
│  └────────┬───────────┘                                              │
│           │ session.save_optimized_model("model_optimized.onnx")?    │
│  ┌────────▼───────────┐                                              │
│  │  Optimization       │    Apply calibration → INT8/INT4 quant       │
│  │  (optional)         │                                              │
│  └────────┬───────────┘                                              │
│           │ Orthogonally: TensorRT EP for GPU, OpenVINO for Intel    │
│  ┌────────▼───────────┐                                              │
│  │  Hardware-Specific  │    TensorRT engine build, kernel auto-tune  │
│  │  Optimization       │                                              │
│  └────────┬───────────┘                                              │
│           │                                                           │
│  ┌────────▼───────────┐                                              │
│  │  Deploy             │    Rust server (Axum/Actix) + Session pool   │
│  │  (Rust production)  │    + Prometheus metrics + health checks      │
│  └────────────────────┘                                              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 🌍 Real-World Applications

| Company | Use Case | Detail |
|---------|----------|--------|
| **Microsoft Azure ML** | Automated model optimization | ONNX Runtime with Level 99 optimization + INT8 quantization applied automatically to customer models |
| **NVIDIA Triton** | ONNX backend with TensorRT | Models pass through ONNX Runtime's optimization stack before TensorRT engine build; custom ops for dynamic batching |
| **HuggingFace Optimum** | Model export and optimization | Optimum exports HuggingFace models to ONNX, applies graph optimizations, and deploys via ONNX Runtime with INT8/INT4 quantization |
| **Tesla Autopilot** | Real-time vision inference | Custom ONNX ops for proprietary vision algorithms; multi-threaded CPU inference with NUMA-aware affinity |
| **Unity Barracuda** | Game engine ML | Custom operators for Unity-specific tensor layouts; Level 2 optimizations to balance performance and determinism |

## ⚠️ Pitfalls

- **Level 3/All can silently break models**: Aggressive layout optimization (NCHW→NHWC) changes tensor memory layout. If your model has custom ops or unsupported patterns, outputs can be silently wrong. Always validate with `onnx.checker` and compare outputs numerically.
- **Custom op ABI stability**: Custom ops link to specific ONNX Runtime C++ ABI versions. Upgrading the `ort` crate can break custom op compatibility. Pin your ORT version in Cargo.toml.
- **spawn_blocking oversubscription**: When using Tokio, if you spawn more blocking tasks than CPU cores, thread contention can increase tail latency. Use a bounded Semaphore to limit concurrent inference calls.
- **INT8 calibration data must match production distribution**: Using random data for calibration produces quantized models that perform well on random inputs but degrade on real data. Always calibrate with representative samples from your actual workload.
- **IO binding with dynamic shapes**: IO binding works best with fixed shapes. Dynamic batches require re-allocating pinned memory buffers, negating the zero-copy benefit.

## 💡 Tips

- Start quantization at INT8 dynamic (no calibration needed) to measure baseline speedup. If accuracy is acceptable, you're done. Only invest in static quantization when dynamic's accuracy is insufficient.
- For multi-GPU setups, create one session pool per GPU and assign requests via consistent hashing on a request ID to maximize cache reuse.
- Profile with `tokio-console` to detect blocking inference calls that starve the async runtime. ONNX Runtime inference is blocking — always use `spawn_blocking`.
- Use `GraphOptimizationLevel::Level2` in production for maximum safety. Reserve Level 3 and All for models that have been validated on representative inputs.
- Pre-compute and cache optimized+quantized models in CI/CD. Session creation with `Level::All` can take 10-60 seconds for large models — this cost should never be paid at runtime.



## 🎯 Key Takeaways

- Graph optimization levels are a safety-vs-performance spectrum: Level 1 is safe, Level 2 is aggressive but validated, Level 3+ can break unvalidated models. Always validate.
- Custom operators extend ONNX Runtime's operator set in Rust, enabling proprietary or research ops without modifying the C++ core. They come with ABI stability risks on `ort` upgrades.
- INT8 quantization offers 4x size reduction and 2-3x speedup. Start with dynamic quantization; use static when accuracy demands better calibration.
- Intra-op (within operator) and inter-op (across branches) parallelism must be tuned together. The sum should not exceed physical cores.
- ONNX Runtime inference is blocking — integrate with Tokio via `spawn_blocking` and bounded semaphores to prevent async runtime starvation.
- Session pooling amortizes session creation cost and enables concurrent inference on multi-GPU setups with hardware-aware routing.

## References
- [[03 - ONNX Runtime Rust|ONNX Runtime Rust]]
- [[04 - ONNX vs Candle: Choosing the Right Runtime|ONNX vs Candle]]
- [[08 - Model Quantization and Optimization|Model Quantization and Optimization]]
- [ONNX Runtime Graph Optimizations](https://onnxruntime.ai/docs/performance/graph-optimizations.html)
- [ONNX Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [HuggingFace Optimum ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/overview)
