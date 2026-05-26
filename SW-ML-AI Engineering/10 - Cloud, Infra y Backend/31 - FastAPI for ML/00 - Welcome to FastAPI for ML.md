# ⚡ Welcome to FastAPI for ML

## 🎯 Learning Objectives

By completing this course, you will be able to:

- Explain why ASGI-based concurrency is essential for production ML inference servers and contrast it with WSGI-based architectures
- Design Pydantic v2 schemas that validate complex ML inputs including nested feature dictionaries, tensor serialization, and batch payloads
- Implement streaming and WebSocket endpoints for real-time ML use cases such as token generation from LLMs and bidirectional speech translation
- Leverage FastAPI's dependency injection system to inject model instances, database sessions, and feature store clients with full testability
- Deploy FastAPI ML services to production with Docker, Kubernetes, TLS termination, and OpenTelemetry-based observability

## Introduction

Machine learning models trained in notebooks represent only a fraction of the engineering effort required to deliver business value. The remaining work spans API design, input validation, concurrency management, observability, and deployment. FastAPI has emerged as the dominant framework for ML serving in Python because it addresses all these concerns within a single, batteries-included library powered by type annotations and async I/O.

Traditional ML deployments often begin with Flask — a synchronous WSGI framework that blocks one worker per request. This architecture collapses under the concurrent load patterns common in ML systems: a single prediction may trigger feature store lookups (see [[../../25 - Bases de Datos y Message Queues/20 - Feature Stores y ML Metadata/20 - Feature Stores y ML Metadata|Feature Stores]]), vector database searches, and post-processing pipelines. FastAPI's ASGI foundation enables a single process to interleave these I/O-bound operations without wasting CPU cycles on blocked workers, a pattern explored in depth in [[../../24 - Backend para ML/00 - Backend para ML|Backend for ML]].

This course bridges the gap between model training and production serving. You will learn to build APIs that validate tensor-shaped inputs with Pydantic v2, stream partial results for LLM token generation (connecting to [[../../30 - WebSockets and Real-Time ML/00 - WebSockets and Real-Time ML|WebSockets and Real-Time ML]]), and deploy containerized services with Kubernetes probes and OpenTelemetry tracing. The patterns covered here extend naturally to deployment strategies discussed in [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/00 - Deployment y Serving|Deployment and Serving]] and testing methodologies from [[../../28 - Testing in ML Systems/00 - Testing in ML Systems|Testing in ML Systems]].

---

## 📋 Course Map

| # | Note | Description | Lines |
|---|------|-------------|-------|
| 01 | ASGI Architecture and Async Python for ML | Event loop mechanics, `async`/`await` deep dive, uvicorn production config, concurrency patterns | ~500 |
| 02 | Pydantic v2 for ML Input/Output Schemas | Rust-core validation, tensor serialization, schema versioning, response models for numpy/torch | ~500 |
| 03 | Streaming, Background Tasks, and Real-Time Endpoints | SSE/StreamingResponse for LLMs, background tasks for logging, file uploads, WebSocket inference | ~500 |
| 04 | Dependency Injection, Middleware, and Testing | `Depends` system, custom middleware for ML metrics, TestClient, lifespan events, health checks | ~500 |
| 05 | Production Deployment and Performance | Traefik/Nginx reverse proxy, Docker/K8s manifests, OpenTelemetry, benchmarking, GPU utilization | ~500 |

## 🧱 Prerequisites

| Topic | Required Proficiency | Related Vault Note |
|-------|---------------------|--------------------|
| Python typing and dataclasses | Intermediate — understand `List[str]`, `Optional`, `Union` | [[../../24 - Backend para ML/01 - Python Typing for ML/01 - Python Typing for ML|Python Typing]] |
| Async Python fundamentals | Basic — `async def`, `await`, coroutine concepts | [[../../24 - Backend para ML/02 - Async Python/02 - Async Python|Async Python]] |
| HTTP and REST concepts | Basic — methods, status codes, headers, JSON | [[../../24 - Backend para ML/00 - Backend para ML|Backend for ML]] |
| Docker fundamentals | Basic — Dockerfile, images, containers | [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/02 - Containerization/02 - Containerization|Containerization]] |
| Pydantic basics | Familiarity with `BaseModel` and `Field` | Covered in Note 02 |

---

## 📦 What You Will Build

By the end of this course, you will construct a production-grade text classification gateway that:

- Accepts batch prediction requests with schema-validated inputs (1–500 texts per call)
- Streams NDJSON responses for large batches to reduce perceived latency
- Provides async endpoints with 202-accepted patterns for heavyweight batches
- Logs every prediction asynchronously via background tasks without blocking responses
- Exposes health-check endpoints and Prometheus metrics for Kubernetes probes
- Runs behind Traefik with TLS termination and rate limiting in a Docker Compose stack

---

## 🔗 Vault Connections

This course sits at the intersection of several knowledge domains:

- **Backend Engineering**: [[../../24 - Backend para ML/00 - Backend para ML|Backend for ML]] — REST API design, middleware, authentication
- **Data Layer**: [[../../25 - Bases de Datos y Message Queues/00 - Bases de Datos y Message Queues|Bases de Datos y Message Queues]] — database sessions, message queue integration, feature stores
- **Real-Time Systems**: [[../../30 - WebSockets and Real-Time ML/00 - WebSockets and Real-Time ML|WebSockets and Real-Time ML]] — bidirectional streaming, SSE, WebSocket patterns
- **MLOps**: [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/00 - Deployment y Serving|Deployment and Serving]] — containerization, orchestration, CI/CD
- **Testing**: [[../../28 - Testing in ML Systems/00 - Testing in ML Systems|Testing in ML Systems]] — unit testing ML endpoints, dependency overrides, integration tests

## References

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [ASGI Specification](https://asgi.readthedocs.io/en/latest/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Starlette Documentation](https://www.starlette.io/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
