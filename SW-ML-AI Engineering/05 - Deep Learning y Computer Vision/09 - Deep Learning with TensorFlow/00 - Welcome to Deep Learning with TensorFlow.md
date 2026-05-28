# 🧠 Deep Learning with TensorFlow

Welcome to the TensorFlow track of the Deep Learning module. This course teaches production-grade deep learning using TensorFlow 2.x, from model architectures to distributed training.

## 📚 Course Roadmap

| Note | Title | Description |
|------|-------|-------------|
| `00` | **Welcome** | Course overview and ecosystem map |
| `01` | **tf.keras Architectures** | Building models with Sequential, Functional, and subclassing APIs |
| `02` | **tf.data and TFRecord Pipelines** | High-performance input pipelines and serialization |
| `03` | **Training at Scale** | Distribution strategies, graph mode, and TPU training |
| `04` | **Model Optimization & Deployment** | Quantization, pruning, TF Serving, and TFLite *(planned)* |
| `05` | **TensorFlow Extended (TFX)** | End-to-end MLOps pipelines *(planned)* |
| `06` | **Advanced Topics** | AutoGraph, custom op kernels, and TF-Agents *(planned)* |

## ✅ Prerequisites

- Solid Python (3.8+) and NumPy proficiency
- Understanding of backpropagation, activation functions, and CNN/RNN basics
- Familiarity with PyTorch is helpful: see [[05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida]]
- Basic Linux/CLI comfort for distributed training sections

## 🔄 TensorFlow 2 vs TensorFlow 1

TF 1.x used static graphs and `Session.run()`. TF 2.x adopted **eager execution by default** and elevated Keras to the official high-level API. This eliminated the boilerplate of `tf.placeholder` and manual graph construction, making TF feel more like PyTorch while retaining graph-mode performance via `tf.function`.

## 🗺️ Ecosystem Map

```
┌─────────────────────────────────────────┐
│           TensorFlow Ecosystem          │
├─────────────┬─────────────┬─────────────┤
│   tf.keras  │   tf.data   │ tf.distribute│
│  (Models)   │ (Pipelines) │   (Scale)   │
├─────────────┴─────────────┴─────────────┤
│      TF Serving  │  TFX  │  TFLite      │
│    (Deployment)  │(MLOps)│ (Mobile/Edge)│
└─────────────────────────────────────────┘
```

- **Keras**: High-level API for rapid prototyping and production models.
- **tf.data**: Scalable, deterministic input pipelines with prefetching and caching.
- **tf.distribute**: Train across GPUs, TPUs, and multiple machines with minimal code changes.
- **TF Serving**: Production-grade REST/gRPC model serving.
- **TFX**: MLOps orchestration for data validation, transformation, training, and monitoring.
- **TFLite**: Optimized inference for mobile, embedded, and edge devices.

## 🤔 When to Choose TensorFlow

| Scenario | Recommendation |
|----------|----------------|
| Google Cloud / TPU workloads | ✅ TensorFlow has first-class TPU support |
| Legacy enterprise codebases | ✅ TF 1→2 migration tools exist; many enterprises standardized on TF |
| Production pipelines requiring SavedModel | ✅ Stable format with TF Serving, TFX, and TFLite |
| Research flexibility / HuggingFace | ✅ PyTorch ecosystem is dominant here |

For deployment patterns, see [[09 - MLOps y Produccion]] and [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome]]. For LLM training, see [[06 - Large Language Models]].

## 🛠️ Environment Setup

```bash
# Recommended: use a virtual environment
python -m venv tf-env
source tf-env/bin/activate

# Install TensorFlow (GPU variant if CUDA is available)
pip install tensorflow==2.15.0

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

Ready? Open [[01 - tf.keras Architectures]] to start building models.
