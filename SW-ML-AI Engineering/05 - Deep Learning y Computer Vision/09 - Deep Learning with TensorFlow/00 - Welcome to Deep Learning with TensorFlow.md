# 🧠 Deep Learning with TensorFlow

Welcome to the TensorFlow track. This course teaches production-grade deep learning using TensorFlow 2.x, from model architectures to distributed training and deployment. The focus is on patterns that scale in real ML systems: `tf.data` pipelines, `tf.distribute` for multi-GPU/TPU training, TensorBoard for observability, SavedModel for deployment, and TFLite for edge inference.

## 📚 Course Map

| Note | Title | What You'll Master |
|------|-------|-------------------|
| `00` | **Welcome** | Course overview and ecosystem map |
| `01` | **tf.keras Architectures** | Sequential, Functional, Subclassing APIs; custom layers/losses/metrics; `train_step` override |
| `02` | **tf.data and TFRecord Pipelines** | Dataset construction, performance optimization, TFRecord serialization, graph-augmented pipelines |
| `03` | **Training at Scale** | `tf.distribute` strategies, `tf.function`/AutoGraph, XLA compilation, distributed checkpointing |
| `04` | **TensorBoard, Callbacks, and Tuning** | Callback lifecycle, built-in and custom callbacks, TensorBoard profiling, KerasTuner |
| `05` | **TensorFlow Production Ecosystem** | SavedModel format, TF Serving (REST/gRPC), TFLite, TF.js, TFX overview |
| `06` | **Capstone — End-to-End CV Pipeline** | Full stack: TFRecord → EfficientNet → MirroredStrategy → Tuning → Serving → Docker Compose |

## ✅ Prerequisites

- Solid Python (3.8+) and NumPy
- Backpropagation, activation functions, CNN/RNN basics
- Familiarity with PyTorch helps, see [[03 - Deep Learning con PyTorch/00 - Bienvenida]]
- Basic CLI comfort for distributed training sections

## 🔄 TF 2 vs TF 1

TF 1.x used static graphs and `Session.run()`. TF 2.x adopted **eager execution by default** and elevated Keras to the official high-level API, eliminating `tf.placeholder` and manual graph construction while retaining graph-mode performance via `tf.function`.

## 🗺️ Ecosystem

- **Keras** — High-level API for rapid prototyping and production models
- **tf.data** — Scalable, deterministic input pipelines with prefetching/caching
- **tf.distribute** — Train across GPUs, TPUs, and multiple machines with minimal code changes
- **TF Serving** — Production-grade REST/gRPC model serving
- **TFX** — MLOps orchestration: data validation, transformation, training, monitoring
- **TFLite** — Optimized inference for mobile, embedded, and edge devices

## 🤔 When to Choose TensorFlow

| Scenario | Verdict |
|----------|---------|
| Google Cloud / TPU workloads | ✅ First-class TPU support |
| Legacy enterprise codebases | ✅ TF 1→2 migration tools; many enterprises standardized on TF |
| Production pipelines requiring SavedModel | ✅ Stable format with TF Serving, TFX, TFLite |
| Research flexibility / HuggingFace | ✅ PyTorch ecosystem is dominant here |

For deployment context see [[09 - MLOps y Produccion]] and [[32 - System Design for ML]]. For LLM training see [[06 - Large Language Models]].

## 🛠️ Environment

```bash
python -m venv tf-env && source tf-env/bin/activate
pip install tensorflow==2.15.0
python -c "import tensorflow as tf; print(tf.__version__)"
```

Ready? Open [[01 - tf.keras Architectures]] to start building models.
