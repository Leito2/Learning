# ⚙️ Welcome to Go for ML Backend

## What will you learn?

This course bridges the gap between Go engineering and machine learning serving. You will learn how to productionize ML models using Go's superior concurrency, memory efficiency, and deployment ergonomics.

- Deploy ONNX models in Go using ONNX Runtime bindings for cross-platform inference
- Build high-throughput model serving systems with batching and worker pools
- Implement feature stores in Go for low-latency feature retrieval
- Design real-time inference pipelines with stream processing platforms
- Evaluate Go vs Python for ML serving with data-driven benchmarks
- Construct production ML gateways with routing, A/B testing, and observability

By the end of this module, you will be capable of designing end-to-end ML backend systems where Go handles serving, orchestration, and scaling, while Python remains in the research and training domain.

## Course Structure

- [[01 - ONNX Runtime Go|🔢 01 - ONNX Runtime]]
- [[02 - High-Throughput Model Serving|🚀 02 - Model Serving]]
- [[03 - Feature Stores with Go|🏪 03 - Feature Stores]]
- [[04 - Real-time Inference Pipelines|⚡ 04 - Inference Pipelines]]
- [[05 - Go vs Python for ML Serving|🥊 05 - Go vs Python]]
- [[06 - Building a Production ML Gateway|🚪 06 - ML Gateway]]

## Capstone Project

Build a **Production ML Inference Platform** in Go that serves an ONNX image classification model. The system must include a feature store backed by Redis, a real-time Kafka pipeline for event-driven inference, an API gateway with canary routing, and Prometheus metrics. Benchmark the system against a Python FastAPI baseline and document the latency, throughput, and memory differences.

---

💡 **Tip:** The best ML systems use Python for research and Go for serving. Master both to build end-to-end pipelines.
