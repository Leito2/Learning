# 📚 Índice Maestro de Cursos — SW-ML-AI Engineering

> **🔁 Reestructuración (Mayo 2026):** El vault ha sido reorganizado con estructura numérica progresiva. Los cursos en inglés antes agrupados en `09 - Extra` ahora están integrados en sus categorías temáticas correspondientes. La navegación es de fundamentos → ML core → producción → especialización → SE stack.
>
> **🔧 Numeración sincronizada (Julio 2026):** El Índice Maestro ahora usa los **mismos prefijos numéricos que las carpetas del filesystem** (00-16). Antes había una desalineación entre las etiquetas lógicas de las secciones (00-08) y los nombres reales de las carpetas (00-16), que rompía los wikilinks. Esta versión corrige todos los enlaces.

Bienvenido al índice completo de tu ruta de aprendizaje como **AI/ML Engineer**. Este vault unifica todos los cursos — desde fundamentos de software engineering hasta sistemas de IA en producción — en un solo ecosistema de conocimiento.

---

## 🗺️ Mapa de la Ruta

```
00 - Curso Markdown (5)
01 - Curso SQL con PostgreSQL (8)
02 - Docker Profesional (7)
03 - Advanced Python (62, 5 módulos)
├── 01 - Python Basico (13)
├── 02 - Python Intermedio (13)
├── 03 - Python Avanzado (13)
├── 04 - Librerias Basicas de Python (8)
└── 05 - Librerias Especificas (7)

04 - Engineering Fundamentals (23)
├── 00 - Python Avanzado para ML (8)
├── 01 - Matematicas para ML (7)
└── 02 - Estructuras de Datos y Algoritmos (8)

05 - Deep Learning y Computer Vision (44)
├── 03 - Deep Learning con PyTorch (7)
├── 04 - Computer Vision Avanzada (6)
├── 05 - Multimodal AI (5)
├── 06 - Computer Vision Pipeline (1 — English)
├── 07 - Reinforcement Learning (7 — English)
├── 08 - Graph Neural Networks (5 — English)
├── 09 - Deep Learning with TensorFlow (7 — English)
├── 10 - JAX Deep Dive (6 — English)
└── 11 - OpenCV (7)

06 - Large Language Models (70)
├── 06 - Fundamentos de LLMs (7)
├── 07 - Fine-Tuning y Adaptación de LLMs (6)
├── 08 - Generación de Texto y Decodificación (6)
├── 09 - Sistemas de LLMs en Producción (6)
├── 10 - Arquitecturas Avanzadas y MoE (5)
├── 11 - Fine-Tuning LLMs (5 — English)
├── 12 - Production RAG (6 — English)
├── 13 - vLLM and Advanced RAG (7 — English)
├── 14 - Unsloth and Efficient Fine-Tuning (5 — English)
├── 15 - LLM Security and Guardrails (5 — English)
├── 16 - HuggingFace Transformers Deep Dive (10 — English)
├── 17 - ColBERT, SGLang and Next-Gen Inference (11 — English)
├── 18 - TensorRT-LLM (3 — English)
├── 19 - LLM Gateway Patterns and LiteLLM (7 — English)
└── 20 - vLLM Production Serving (9)

07 - AI Agents y Agentic Systems (40)
├── 11 - Fundamentos de Agentes AI (6)
├── 12 - Frameworks y Orquestación (6)
├── 13 - Sistemas Multi-Agente (5)
├── 14 - Agentes Autónomos y Auto-Mejora (5)
├── 15 - MCP and Agentic Protocols (6 — English)
├── 16 - OpenShell and Agent Sandboxes (6 — English)
└── 17 - Production Agent Frameworks (8 — English, NEW)

08 - NLP Avanzado (17)
├── 15 - NLP Tradicional y Representaciones (6)
├── 16 - NLP con Transformers (6)
└── 17 - NLP Aplicado e Industria (5)

09 - MLOps y Produccion (59)
├── 18 - Experiment Tracking y Model Registry (7)
├── 19 - Feature Engineering y Feature Stores (6)
├── 20 - Deployment y Serving (6)
├── 21 - Monitoreo y Mantenimiento (6)
├── 22 - End-to-End ML Project (5 — English)
├── 23 - Advanced MLOps (1 — English)
├── 24 - Weights and Biases (5 — English)
├── 25 - MLOps Tooling Comparison (1 — English)
├── 26 - ML Platform Engineering (6 — English)
├── 27 - Feast and Feature Stores (5 — English)
├── 28 - Testing in ML Systems (4 — English)
├── 29 - CI-CD for ML (4 — English)
├── 30 - TorchServe (4 — English)
├── 31 - Evidently AI and Phoenix (4 — English)
├── 32 - KServe and Knative (3 — English)
└── 33 - Temporal for ML Pipelines (3 — English)

10 - Cloud, Infra y Backend (64)
├── 22 - Cloud Computing (6)
├── 23 - Infrastructure as Code (10 — English)
├── 24 - (merged into 31)
├── 25 - Bases de Datos y Message Queues (6)
├── 26 - Databricks for ML (5 — English)
├── 27 - Apache Spark for ML (6 — English)
├── 28 - BigQuery for ML (3 — English)
├── 29 - Distributed ML Infrastructure (7 — English)
├── 30 - WebSockets and Real-Time ML (5 — English)
├── 31 - FastAPI for ML (11 — English, merged with 24)
├── 32 - System Design for ML (6 — English)
├── 33 - Vector Databases and Semantic Search (17 — English)
├── 34 - DuckDB (4 — English)
├── 35 - Vector Quantization and Approximate Nearest Neighbors (6 — English)
├── 36 - PostgreSQL for AI-ML Workloads (6 — English)
├── 37 - Vector Search on Google Cloud (3 — English)
├── 38 - SQLAlchemy 2.0 Async + Alembic for FastAPI (7 — English)
├── 39 - Authentication Deep Dive for FastAPI (6 — English)
├── 40 - Background Jobs and Workers for FastAPI (5 — English)
├── 41 - API Design Patterns for FastAPI (5 — English)
├── 42 - Caching Strategies for FastAPI (4 — English)
├── 43 - File Storage and Uploads for FastAPI (4 — English)
├── 44 - Email and Notifications for FastAPI (3 — English)
└── 45 - Webhooks In and Out for FastAPI (3 — English)

11 - Research y Ciencia de Datos (33)
├── 26 - Metodología de Investigación en ML (6)
├── 27 - Visualización de Datos y Storytelling (6)
├── 28 - ETL y Data Engineering (6)
├── 29 - Estadística Avanzada y Causalidad (6)
├── 30 - Kaggle Competitions (1 — English)
├── 31 - Paper Reproduction (1 — English)
└── 32 - Advanced ML Topics (7 — English)

12 - Producto, Negocio y Open Source (18)
├── 30 - Producto y Estrategia de IA (6)
├── 31 - Negocio y Métricas de ML (6)
└── 32 - Open Source y Comunidad (6)

13 - Go Engineering (73+, 9 cursos + extra + projects)
14 - Rust Engineering (73+, 8 cursos + extra + projects)
15 - Transversal Skills (4 — English)
16 - Harness Engineering (10 — English)

Proyectos
└── projects/ (14 guías: ML + SE)

Misceláneo
└── Extra/
```

---

## 📚 Cursos — Detalle por Sección

### 00 — Curso Markdown (5 notas)

| Curso | Notas | Estado | Idioma |
|-------|:-----:|:------:|:------:|
| [[00 - Curso Markdown/00 - Bienvenida al Curso Markdown\|Curso Comprimido de Markdown]] | 5 | ✅ | 🇪🇸 |

### 01 — Curso SQL con PostgreSQL (8 notas)

| Curso | Notas | Estado | Idioma |
|-------|:-----:|:------:|:------:|
| [[01 - Curso SQL con PostgreSQL/00 - Bienvenida al Curso SQL\|Curso SQL con PostgreSQL]] | 8 | ✅ | 🇪🇸 |

### 02 — Docker Profesional (7 notas)

| Curso | Notas | Estado | Idioma |
|-------|:-----:|:------:|:------:|
| [[02 - Docker Profesional/00 - Bienvenida\|Docker Profesional]] | 7 | ✅ | 🇪🇸 |

### 03 — Advanced Python (62 notas, 5 módulos)

| # | Módulo | Notas | Estado | Idioma |
|---|--------|:-----:|:------:|:------:|
| 01 | [[03 - Advanced Python/01 - Python Basico/00 - Bienvenida\|Python Básico]] | 13 | ✅ | 🇪🇸 |
| 02 | [[03 - Advanced Python/02 - Python Intermedio/00 - Bienvenida\|Python Intermedio]] | 13 | ✅ | 🇪🇸 |
| 03 | [[03 - Advanced Python/03 - Python Avanzado/00 - Bienvenida\|Python Avanzado]] | 13 | ✅ | 🇪🇸 |
| 04 | [[03 - Advanced Python/04 - Librerias Basicas de Python/00 - Bienvenida\|Librerías Básicas]] | 8 | ✅ | 🇪🇸 |
| 05 | [[03 - Advanced Python/05 - Librerias Especificas/00 - Bienvenida\|Librerías Específicas]] | 7 | ✅ | 🇪🇸 |

### 04 — Engineering Fundamentals (23 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 00 | [[04 - Engineering Fundamentals/00 - Python Avanzado para ML/00 - Bienvenida\|Python Avanzado para ML]] | 8 | ✅ | 🇪🇸 |
| 01 | [[04 - Engineering Fundamentals/01 - Matematicas para ML/00 - Bienvenida\|Matemáticas para ML]] | 7 | ✅ | 🇪🇸 |
| 02 | [[04 - Engineering Fundamentals/02 - Estructuras de Datos y Algoritmos/00 - Bienvenida\|Estructuras de Datos y Algoritmos]] | 8 | ✅ | 🇪🇸 |

### 05 — Deep Learning y Computer Vision (44 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 03 | [[05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida\|Deep Learning con PyTorch]] | 7 | ✅ | 🇪🇸 |
| 04 | [[05 - Deep Learning y Computer Vision/04 - Computer Vision Avanzada/00 - Bienvenida\|Computer Vision Avanzada]] | 6 | ✅ | 🇪🇸 |
| 05 | [[05 - Deep Learning y Computer Vision/05 - Multimodal AI/00 - Bienvenida\|Multimodal AI]] | 5 | ✅ | 🇪🇸 |
| 06 | [[05 - Deep Learning y Computer Vision/06 - Computer Vision Pipeline/05 - Computer Vision Pipeline\|Computer Vision Pipeline]] | 1 | ✅ | 🇬🇧 |
| 07 | [[05 - Deep Learning y Computer Vision/07 - Reinforcement Learning/00 - Welcome to RL\|Reinforcement Learning]] | 7 | ✅ | 🇬🇧 |
| 08 | [[05 - Deep Learning y Computer Vision/08 - Graph Neural Networks/00 - Welcome to GNN\|Graph Neural Networks]] | 5 | ✅ | 🇬🇧 |
| 09 | [[05 - Deep Learning y Computer Vision/09 - Deep Learning with TensorFlow/00 - Welcome to Deep Learning with TensorFlow\|Deep Learning with TensorFlow]] | 7 | ✅ | 🇬🇧 |
| 10 | [[05 - Deep Learning y Computer Vision/10 - JAX Deep Dive/00 - Welcome to JAX Deep Dive\|JAX Deep Dive]] | 6 | ✅ NEW | 🇬🇧 |
| 11 | [[05 - Deep Learning y Computer Vision/11 - OpenCV/00 - Bienvenida\|OpenCV]] | 7 | ✅ NEW | 🇪🇸 |

### 06 — Large Language Models (70 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 06 | [[06 - Large Language Models/06 - Fundamentos de LLMs/00 - Bienvenida\|Fundamentos de LLMs]] | 7 | ✅ | 🇪🇸 |
| 07 | [[06 - Large Language Models/07 - Fine-Tuning y Adaptacion de LLMs/00 - Bienvenida\|Fine-Tuning y Adaptación de LLMs]] | 6 | ✅ | 🇪🇸 |
| 08 | [[06 - Large Language Models/08 - Generacion de Texto y Decodificacion/00 - Bienvenida\|Generación de Texto y Decodificación]] | 6 | ✅ | 🇪🇸 |
| 09 | [[06 - Large Language Models/09 - Sistemas de LLMs en Produccion/00 - Bienvenida\|Sistemas de LLMs en Producción]] | 6 | ✅ | 🇪🇸 |
| 10 | [[06 - Large Language Models/10 - Arquitecturas Avanzadas y MoE/00 - Bienvenida\|Arquitecturas Avanzadas y MoE]] | 5 | ✅ | 🇪🇸 |
| 11 | [[06 - Large Language Models/11 - Fine-Tuning LLMs/00 - Welcome to Fine-Tuning LLMs\|Fine-Tuning LLMs]] | 5 | ✅ REWRITTEN | 🇬🇧 |
| 12 | [[06 - Large Language Models/12 - Production RAG/00 - Welcome to Production RAG\|Production RAG]] | 6 | ✅ REWRITTEN | 🇬🇧 |
| 13 | [[06 - Large Language Models/13 - vLLM and Advanced RAG/00 - Welcome to vLLM and Advanced RAG\|vLLM and Advanced RAG]] | 7 | ✅ | 🇬🇧 |
| 14 | [[06 - Large Language Models/14 - Unsloth and Efficient Fine-Tuning/00 - Welcome to Unsloth and Efficient Fine-Tuning\|Unsloth and Efficient Fine-Tuning]] | 5 | ✅ | 🇬🇧 |
| 15 | [[06 - Large Language Models/15 - LLM Security and Guardrails/00 - Welcome to LLM Security and Guardrails\|LLM Security and Guardrails]] | 5 | ✅ | 🇬🇧 |
| 16 | [[06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/00 - Welcome to HuggingFace Transformers Deep Dive\|HuggingFace Transformers Deep Dive]] | 10 | ✅ NEW | 🇬🇧 |
| 17 | [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference/00 - Welcome to ColBERT, SGLang and Next-Gen Inference\|ColBERT, SGLang and Next-Gen Inference]] | 11 | ✅ REWRITTEN | 🇬🇧 |
| 18 | [[06 - Large Language Models/18 - TensorRT-LLM/00 - Welcome to TensorRT-LLM\|TensorRT-LLM]] | 3 | ✅ NEW | 🇬🇧 |
| 19 | [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns and LiteLLM]] | 7 | ✅ NEW | 🇬🇧 |
| 20 | [[06 - Large Language Models/20 - vLLM Production Serving/00 - Bienvenida\|vLLM Production Serving]] | 9 | ✅ NEW | 🇪🇸 |

### 07 — AI Agents y Agentic Systems (40 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 11 | [[07 - AI Agents y Agentic Systems/11 - Fundamentos de Agentes AI/00 - Bienvenida\|Fundamentos de Agentes AI]] | 6 | ✅ | 🇪🇸 |
| 12 | [[07 - AI Agents y Agentic Systems/12 - Frameworks y Orquestacion/00 - Bienvenida\|Frameworks y Orquestación]] | 6 | ✅ | 🇪🇸 |
| 13 | [[07 - AI Agents y Agentic Systems/13 - Sistemas Multi-Agente/00 - Bienvenida\|Sistemas Multi-Agente]] | 5 | ✅ | 🇪🇸 |
| 14 | [[07 - AI Agents y Agentic Systems/14 - Agentes Autonomos y Auto-Mejora/00 - Bienvenida\|Agentes Autónomos y Auto-Mejora]] | 5 | ✅ | 🇪🇸 |
| 15 | [[07 - AI Agents y Agentic Systems/15 - MCP and Agentic Protocols/00 - Welcome to MCP and Agentic Protocols\|MCP and Agentic Protocols]] | 6 | ✅ | 🇬🇧 |
| 16 | [[07 - AI Agents y Agentic Systems/16 - OpenShell and Agent Sandboxes/00 - Welcome to OpenShell and Agent Sandboxes\|OpenShell and Agent Sandboxes]] | 6 | ✅ NEW | 🇬🇧 |
| 17 | [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks/00 - Welcome to Production Agent Frameworks\|Production Agent Frameworks]] | 8 | ✅ NEW | 🇬🇧 |

### 08 — NLP Avanzado (17 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 15 | [[08 - NLP Avanzado/15 - NLP Tradicional y Representaciones/00 - Bienvenida\|NLP Tradicional y Representaciones]] | 6 | ✅ | 🇪🇸 |
| 16 | [[08 - NLP Avanzado/16 - NLP con Transformers/00 - Bienvenida\|NLP con Transformers]] | 6 | ✅ | 🇪🇸 |
| 17 | [[08 - NLP Avanzado/17 - NLP Aplicado e Industria/00 - Bienvenida\|NLP Aplicado e Industria]] | 5 | ✅ | 🇪🇸 |

### 09 — MLOps y Producción (59 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 18 | [[09 - MLOps y Produccion/18 - Experiment Tracking y Model Registry/00 - Bienvenida\|Experiment Tracking y Model Registry]] | 7 | ✅ | 🇪🇸 |
| 19 | [[09 - MLOps y Produccion/19 - Feature Engineering y Feature Stores/00 - Bienvenida\|Feature Engineering y Feature Stores]] | 6 | ✅ | 🇪🇸 |
| 20 | [[09 - MLOps y Produccion/20 - Deployment y Serving/00 - Bienvenida\|Deployment y Serving]] | 6 | ✅ | 🇪🇸 |
| 21 | [[09 - MLOps y Produccion/21 - Monitoreo y Mantenimiento/00 - Bienvenida\|Monitoreo y Mantenimiento]] | 6 | ✅ | 🇪🇸 |
| 22 | [[09 - MLOps y Produccion/22 - End-to-End ML Project/00 - Welcome to End-to-End ML Project\|End-to-End ML Project]] | 5 | ✅ REWRITTEN | 🇬🇧 |
| 28 | [[09 - MLOps y Produccion/28 - Testing in ML Systems/00 - Welcome to Testing in ML Systems\|Testing in ML Systems]] | 4 | ✅ REWRITTEN | 🇬🇧 |
| 29 | [[09 - MLOps y Produccion/29 - CI-CD for ML/00 - Welcome to CI-CD for ML\|CI-CD for ML]] | 4 | ✅ REWRITTEN | 🇬🇧 |
| 30 | [[09 - MLOps y Produccion/30 - TorchServe/00 - Welcome to TorchServe\|TorchServe]] | 4 | ✅ NEW | 🇬🇧 |
| 31 | [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix/00 - Welcome to Evidently AI and Phoenix\|Evidently AI and Phoenix]] | 4 | ✅ NEW | 🇬🇧 |
| 32 | [[09 - MLOps y Produccion/32 - KServe and Knative/00 - Welcome to KServe and Knative\|KServe and Knative]] | 3 | ✅ NEW | 🇬🇧 |
| 33 | [[09 - MLOps y Produccion/33 - Temporal for ML Pipelines/00 - Welcome to Temporal for ML Pipelines\|Temporal for ML Pipelines]] | 3 | ✅ NEW | 🇬🇧 |

### 10 — Cloud, Infra y Backend (64 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 22 | [[10 - Cloud, Infra y Backend/22 - Cloud Computing/00 - Bienvenida\|Cloud Computing]] | 6 | ✅ | 🇪🇸 |
| 23 | [[10 - Cloud, Infra y Backend/23 - Infrastructure as Code/00 - Welcome to Infrastructure as Code\|Infrastructure as Code]] | 10 | ✅ REWRITTEN | 🇬🇧 |
| 24 | Merged into [[10 - Cloud, Infra y Backend/31 - FastAPI for ML/00 - Welcome\|FastAPI for ML]] | — | ✅ MERGED | 🇪🇸→🇬🇧 |
| 25 | [[10 - Cloud, Infra y Backend/25 - Bases de Datos y Message Queues/00 - Bienvenida\|Bases de Datos y Message Queues]] | 6 | ✅ | 🇪🇸 |
| 26 | [[10 - Cloud, Infra y Backend/26 - Databricks for ML/00 - Welcome to Databricks for ML\|Databricks for ML]] | 5 | ✅ | 🇬🇧 |
| 27 | [[10 - Cloud, Infra y Backend/27 - Apache Spark for ML/00 - Welcome to Apache Spark for ML\|Apache Spark for ML]] | 6 | ✅ | 🇬🇧 |
| 28 | [[10 - Cloud, Infra y Backend/28 - BigQuery for ML/00 - Welcome to BigQuery for ML\|BigQuery for ML]] | 3 | ✅ | 🇬🇧 |
| 29 | [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome\|Distributed ML Infrastructure]] | 7 | ✅ | 🇬🇧 |
| 30 | [[10 - Cloud, Infra y Backend/30 - WebSockets and Real-Time ML/00 - Welcome to WebSockets and Real-Time ML\|WebSockets and Real-Time ML]] | 5 | ✅ | 🇬🇧 |
| 31 | [[10 - Cloud, Infra y Backend/31 - FastAPI for ML/00 - Welcome\|FastAPI for ML]] | 11 | ✅ MERGED (24→31) | 🇬🇧 |
| 32 | [[10 - Cloud, Infra y Backend/32 - System Design for ML/00 - Welcome to System Design for ML\|System Design for ML]] | 6 | ✅ REWRITTEN | 🇬🇧 |
| 33 | [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search\|Vector Databases and Semantic Search]] | 17 | ✅ +Pinecone + Qdrant Python | 🇬🇧 |
| 34 | [[10 - Cloud, Infra y Backend/34 - DuckDB/00 - Welcome to DuckDB\|DuckDB]] | 4 | ✅ NEW | 🇬🇧 |
| 35 | [[10 - Cloud, Infra y Backend/35 - Vector Quantization and Approximate Nearest Neighbors/00 - Welcome to Vector Quantization and Approximate Nearest Neighbors\|Vector Quantization and ANN]] | 6 | ✅ NEW | 🇬🇧 |
| 36 | [[10 - Cloud, Infra y Backend/36 - PostgreSQL for AI-ML Workloads/00 - Welcome to PostgreSQL for AI-ML Workloads\|PostgreSQL for AI-ML Workloads]] | 6 | ✅ NEW | 🇬🇧 |
| 37 | [[10 - Cloud, Infra y Backend/37 - Vector Search on Google Cloud/00 - Welcome to Vector Search on Google Cloud\|Vector Search on Google Cloud]] | 3 | ✅ NEW | 🇬🇧 |
| 38 | [[10 - Cloud, Infra y Backend/38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/00 - Welcome\|SQLAlchemy 2.0 Async + Alembic]] | 7 | ✅ NEW | 🇬🇧 |
| 39 | [[10 - Cloud, Infra y Backend/39 - Authentication Deep Dive for FastAPI/00 - Welcome\|Authentication Deep Dive]] | 6 | ✅ NEW | 🇬🇧 |
| 40 | [[10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI/00 - Welcome\|Background Jobs and Workers]] | 5 | ✅ NEW | 🇬🇧 |
| 41 | [[10 - Cloud, Infra y Backend/41 - API Design Patterns for FastAPI/00 - Welcome\|API Design Patterns]] | 5 | ✅ NEW | 🇬🇧 |
| 42 | [[10 - Cloud, Infra y Backend/42 - Caching Strategies for FastAPI/00 - Welcome\|Caching Strategies]] | 4 | ✅ NEW | 🇬🇧 |
| 43 | [[10 - Cloud, Infra y Backend/43 - File Storage and Uploads for FastAPI/00 - Welcome\|File Storage and Uploads]] | 4 | ✅ NEW | 🇬🇧 |
| 44 | [[10 - Cloud, Infra y Backend/44 - Email and Notifications for FastAPI/00 - Welcome\|Email and Notifications]] | 3 | ✅ NEW | 🇬🇧 |
| 45 | [[10 - Cloud, Infra y Backend/45 - Webhooks In and Out for FastAPI/00 - Welcome\|Webhooks In/Out]] | 3 | ✅ NEW | 🇬🇧 |

### 11 — Research y Ciencia de Datos (33 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 26 | [[11 - Research y Ciencia de Datos/26 - Metodologia de Investigacion en ML/00 - Bienvenida\|Metodología de Investigación en ML]] | 6 | ✅ | 🇪🇸 |
| 27 | [[11 - Research y Ciencia de Datos/27 - Visualizacion de Datos y Storytelling/00 - Bienvenida\|Visualización de Datos y Storytelling]] | 6 | ✅ | 🇪🇸 |
| 28 | [[11 - Research y Ciencia de Datos/28 - ETL y Data Engineering/00 - Bienvenida\|ETL y Data Engineering]] | 6 | ✅ | 🇪🇸 |
| 29 | [[11 - Research y Ciencia de Datos/29 - Estadistica Avanzada y Causalidad/00 - Bienvenida\|Estadística Avanzada y Causalidad]] | 6 | ✅ | 🇪🇸 |
| 30 | [[11 - Research y Ciencia de Datos/30 - Kaggle Competitions/01 - Kaggle Competitions\|Kaggle Competitions]] | 1 | ✅ | 🇬🇧 |
| 31 | [[11 - Research y Ciencia de Datos/31 - Paper Reproduction/07 - Paper Reproduction\|Paper Reproduction]] | 1 | ✅ | 🇬🇧 |
| 32 | [[11 - Research y Ciencia de Datos/32 - Advanced ML Topics/01 - JAX and Flax\|Advanced ML Topics]] | 7 | ✅ | 🇬🇧 |

### 12 — Producto, Negocio y Open Source (18 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 30 | [[12 - Producto, Negocio y Open Source/30 - Producto y Estrategia de IA/00 - Bienvenida\|Producto y Estrategia de IA]] | 6 | ✅ | 🇪🇸 |
| 31 | [[12 - Producto, Negocio y Open Source/31 - Negocio y Metricas de ML/00 - Bienvenida\|Negocio y Métricas de ML]] | 6 | ✅ | 🇪🇸 |
| 32 | [[12 - Producto, Negocio y Open Source/32 - Open Source y Comunidad/00 - Bienvenida\|Open Source y Comunidad]] | 6 | ✅ | 🇪🇸 |

### 13 — Go Engineering (73+ notas, 9 cursos + extra + projects)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 01 | [[13 - Go Engineering/01 - Go Fundamentals/00 - Welcome\|Go Fundamentals]] | 7 | ✅ Deep format | 🇬🇧 |
| 02 | [[13 - Go Engineering/02 - Go for Cloud Native/00 - Welcome\|Go for Cloud Native]] | 7 | ✅ | 🇬🇧 |
| 03 | [[13 - Go Engineering/03 - Microservices with Go/00 - Welcome\|Microservices with Go]] | 7 | ✅ | 🇬🇧 |
| 04 | [[13 - Go Engineering/04 - DevSecOps and CLI Tools/00 - Welcome\|DevSecOps and CLI Tools]] | 7 | ✅ | 🇬🇧 |
| 05 | [[13 - Go Engineering/05 - Local AI with Go/00 - Welcome\|Local AI with Go]] | 7 | ✅ | 🇬🇧 |
| 06 | [[13 - Go Engineering/06 - Go for ML Backend/00 - Welcome\|Go for ML Backend]] | 7 | ✅ | 🇬🇧 |
| 07a | [[13 - Go Engineering/07 - Gorgonia - Deep Learning in Go/00 - Welcome to Gorgonia\|Gorgonia — Deep Learning in Go]] | 6 | ✅ NEW | 🇬🇧 |
| 07b | [[13 - Go Engineering/07 - LocalAI - Local LLM Server/00 - Welcome to LocalAI\|LocalAI — Local LLM Server]] | 6 | ✅ NEW | 🇬🇧 |
| 07c | [[13 - Go Engineering/07 - Wails - Desktop Apps with Go/00 - Welcome to Wails\|Wails — Desktop Apps with Go]] | 6 | ✅ NEW | 🇬🇧 |
| — | [[13 - Go Engineering/extra/00 - Welcome to Go Extra\|Go Extra]] | 7 | ✅ | 🇬🇧 |
| — | [[13 - Go Engineering/projects/00 - Go Project Planning Guide\|Go Projects]] | 6 | ✅ | 🇬🇧 |

### 14 — Rust Engineering (73+ notas, 8 cursos + extra + projects)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 01 | [[14 - Rust Engineering/01 - Rust Fundamentals/00 - Welcome\|Rust Fundamentals]] | 7 | ✅ Deep format | 🇬🇧 |
| 02 | [[14 - Rust Engineering/02 - Advanced Rust/00 - Welcome\|Advanced Rust]] | 7 | ✅ | 🇬🇧 |
| 03 | [[14 - Rust Engineering/03 - Rust for Data Engineering/00 - Welcome\|Rust for Data Engineering]] | 7 | ✅ | 🇬🇧 |
| 04 | [[14 - Rust Engineering/04 - Rust for ML and AI/00 - Welcome\|Rust for ML and AI]] | 7 | ✅ | 🇬🇧 |
| 05 | [[14 - Rust Engineering/05 - WebAssembly and Edge AI/00 - Welcome\|WebAssembly and Edge AI]] | 7 | ✅ | 🇬🇧 |
| 06 | [[14 - Rust Engineering/06 - Agentic AI with Rust/00 - Welcome\|Agentic AI with Rust]] | 7 | ✅ | 🇬🇧 |
| 07a | [[14 - Rust Engineering/07 - Polars Internals and Advanced/00 - Welcome to Polars Internals\|Polars Internals and Advanced]] | 6 | ✅ NEW | 🇬🇧 |
| 07b | [[14 - Rust Engineering/07 - Candle Advanced Patterns/00 - Welcome to Candle Advanced Patterns\|Candle Advanced Patterns]] | 6 | ✅ NEW | 🇬🇧 |
| — | [[14 - Rust Engineering/extra/00 - Welcome to Rust Extra\|Rust Extra]] | 7 | ✅ | 🇬🇧 |
| — | [[14 - Rust Engineering/projects/00 - Rust Project Planning Guide\|Rust Projects]] | 6 | ✅ | 🇬🇧 |

### 15 — Transversal Skills (4 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 01 | [[15 - Transversal Skills/01 - Technical English\|Technical English]] | 1 | ✅ | 🇬🇧 |
| 02 | [[15 - Transversal Skills/02 - Technical Communication and Storytelling\|Technical Communication and Storytelling]] | 1 | ✅ | 🇬🇧 |
| 03 | [[15 - Transversal Skills/03 - Technical Leadership\|Technical Leadership]] | 1 | ✅ | 🇬🇧 |

### 16 — Harness Engineering (10 notas)

| # | Curso | Notas | Estado | Idioma |
|---|-------|:-----:|:------:|:------:|
| 00 | [[16 - Harness Engineering/00 - Welcome to Harness Engineering and SDD\|Harness Engineering and SDD]] | 1 | ✅ | 🇬🇧 |
| 01 | [[16 - Harness Engineering/01 - The Context Crisis\|The Context Crisis]] | 1 | ✅ | 🇬🇧 |
| 02 | [[16 - Harness Engineering/02 - The Three Pillars\|The Three Pillars]] | 1 | ✅ | 🇬🇧 |
| 03 | [[16 - Harness Engineering/03 - Harness Engineering - Architecture of Control\|Harness Engineering Architecture]] | 1 | ✅ | 🇬🇧 |
| 04 | [[16 - Harness Engineering/04 - Specification-Driven Development\|Specification-Driven Development]] | 1 | ✅ | 🇬🇧 |
| 05 | [[16 - Harness Engineering/05 - File Architecture\|File Architecture]] | 1 | ✅ | 🇬🇧 |
| 06 | [[16 - Harness Engineering/06 - Multi-Agent Orchestration and Capstone\|Multi-Agent Orchestration and Capstone]] | 1 | ✅ | 🇬🇧 |
| 07 | [[16 - Harness Engineering/07 - Complete Harness Taxonomy\|Complete Harness Taxonomy]] | 1 | ✅ | 🇬🇧 |
| 08 | [[16 - Harness Engineering/08 - Verification and Quality Gates\|Verification and Quality Gates]] | 1 | ✅ | 🇬🇧 |
| 09 | [[16 - Harness Engineering/09 - Tools, Provider Abstraction, and Memory\|Tools, Provider Abstraction, and Memory]] | 1 | ✅ | 🇬🇧 |

---

## 🎯 Proyectos (14 guías)

### ML Projects

| # | Proyecto |
|---|----------|
| 00 | [[projects/00 - Project Planning Guide for ML\|Project Planning Guide for ML]] |
| 01 | [[projects/01 - Kaggle Competitions - Project Guide\|Kaggle Competitions]] |
| 02 | [[projects/02 - End-to-End ML Project - Project Guide\|End-to-End ML Project]] |
| 03 | [[projects/03 - Fine-Tuning LLMs - Project Guide\|Fine-Tuning LLMs]] |
| 04 | [[projects/04 - Production RAG System - Project Guide\|Production RAG System]] |
| 05 | [[projects/05 - Computer Vision Pipeline - Project Guide\|Computer Vision Pipeline]] |
| 06 | [[projects/06 - Advanced MLOps - Project Guide\|Advanced MLOps]] |
| 07 | [[projects/07 - Paper Reproduction - Project Guide\|Paper Reproduction]] |

### Software Engineering Projects

| # | Proyecto |
|---|----------|
| 00 | [[projects/00 - Project Planning Guide for SE\|Project Planning Guide for SE]] |
| 01 | [[projects/01 - FastAPI for ML - Project Guide\|FastAPI for ML]] |
| 02 | [[projects/02 - System Design for ML - Project Guide\|System Design for ML]] |
| 03 | [[projects/03 - Testing in ML Systems - Project Guide\|Testing in ML Systems]] |
| 04 | [[projects/04 - CI-CD for ML - Project Guide\|CI-CD for ML]] |
| 05 | [[projects/05 - Rust for ML Infra - Project Guide\|Rust for ML Infra]] |

---

## 📊 Skills Maps

| Archivo | Descripción |
|---------|-------------|
| [[Skills Tree - The AI-ML Engineer Growth Map\|Skills Tree]] | Career growth map + hiring analysis |

---

## 🎯 Progresión Recomendada

```
Fase 1 (Meses 1-2):    00-03 — Tooling (Markdown, SQL, Docker, Python)
Fase 2 (Meses 3-4):    04-05 — Engineering Fundamentals + Deep Learning/CV
Fase 3 (Meses 5-6):    06-07 — Large Language Models + AI Agents
Fase 4 (Meses 7-9):    08-10 — NLP Avanzado + MLOps + Cloud/Infra
Fase 5 (Meses 10-12):  11-12 — Research + Producto/Negocio
Fase 6 (Paralelo):     13-14 — Go Engineering + Rust Engineering
Fase 7 (Contínuo):     15-16 — Transversal Skills + Harness Engineering + Proyectos + Portfolio
```

---

## 📈 Estadísticas del Vault

| Métrica | Valor |
|---------|:-----:|
| Cursos totales | 61 |
| Cursos en Español | 26 |
| Cursos en English | 30 |
| Módulos numerados (00-16) | 17 |
| Cursos Go | 11 |
| Cursos Rust | 10 |
| Notas totales | 619+ |
| Guías de proyecto | 14 |

---

> **Nota:** Cada curso incluye teoría profunda, código de ejemplo y un proyecto documentado. Avanza a tu propio ritmo y marca los completados.
>
> Para planificar adiciones futuras, consulta [[Continuity Prompt]].
