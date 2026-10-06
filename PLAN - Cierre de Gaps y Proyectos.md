# 🗺️ PLAN — Cierre de Gaps + Proyectos de Portafolio (P0–P4)

> **Documento temporal.** Vive en la raíz de `Learning` mientras se implementa. Se borra al terminar todos los cursos del vault.
> Los planes de cada proyecto viven en sus repos, como `PLAN.md`, dentro de `Documents/AI Engineer proyects/` (`llm-gateway`, `realtime-fraud-detection`, `smart-request-router`, `live-rag-platform`, `graphrag-research-agents`). Todos públicos en `github.com/Leito2/`.
> **Estado:** 🟡 En planeación · **Creado:** 2026-10-05

---

## 0. Contexto y reglas del juego

### Objetivo
Cerrar los gaps del stack de **ML en tiempo real + LLMOps** que aparecen en el roadmap:

| Capa | Tecnologías objetivo |
|---|---|
| Ingesta / mensajería | Kafka |
| Procesamiento de streams | Apache Flink (+ Spark Structured Streaming, Quix Streams como contraste) |
| Feature store en tiempo real | Redis |
| Serving de modelos | Triton, TorchServe, vLLM (+ ONNX Runtime) |
| Transporte de baja latencia | gRPC, WebSockets, SSE (streaming de tokens) |
| Bases vectoriales | Qdrant, pgvector |
| Observabilidad | Grafana, OpenTelemetry (+ Prometheus, Langfuse) |
| Modelos de decisión | XGBoost · Decision models tipo Jev (Laya) · LLM (Haiku) |

Además, respaldar el CV con **5 proyectos medibles** (P0 gateway compartido + P1–P4) que integren estos gaps, las skills más fuertes del CV y **todo lo que demuestran los 3 proyectos actuales del CV**, que quedan aparte (ver §8).

### Restricciones (no negociables)
| Regla | Detalle |
|---|---|
| 💻 Hardware | **i5-10300H (4C/8T), 8 GB de RAM y 4 GB de VRAM.** La CPU será probablemente el límite de throughput. Todo debe funcionar así. Lo que necesite más queda marcado como `⏳ 16GB` |
| 🐳 Ejecución | **Local con Docker Compose** como camino principal. Servicios cloud gratis (Langfuse Cloud, Qdrant Cloud free) se pueden usar |
| ☁️ Cloud | **Solo P3 en GCP**, con Terraform para crear y destruir, alerta de presupuesto antes del primer deploy y nada que cobre en reposo |
| 💸 Gasto | **$0 hasta el testeo final.** El LLM se llama vía mock o local (Ollama) durante el desarrollo. Haiku se usa **solo en P3** y **solo en la prueba final** |
| 📏 Honestidad | Las metas (20k ev/s, p95 de 80 ms) son objetivos, no resultados garantizados. Al CV va la cifra medida, con su hardware declarado |
| 🌐 Idioma | Cursos nuevos del vault en **inglés** (Language Policy del Continuity Prompt). Planes en español. READMEs de proyectos en inglés (portafolio) |
| 📐 Formato | Las notas siguen el **Deep Format** del `Continuity Prompt.md`, con profundidad adaptativa y el estilo de "Course Design Patterns" |

### Fuera de alcance (decidido)
Linux administration, JAX, TensorFlow, Azure y Kubernetes en los proyectos (ya hay cursos de K8s; aquí todo es Compose). **Sin AWS ni SageMaker** (decisión del usuario, 2026-10-06): la evaluación batch corre en contenedores propios con KFP local sobre MinIO (API S3).

---

## 1. Inventario: qué ya cubre el vault

| Tema | Estado | Dónde | Acción |
|---|---|---|---|
| Kafka | ✅ | `10/29/01`, `10/25/04`, `09/40/*` | Reutilizar y enlazar |
| Flink | ❌ | Solo menciones sueltas | **Curso nuevo (C1)** |
| Spark Structured Streaming | 🟡 | `10/27/04` | Revisar y ampliar si P3 lo requiere (en C2) |
| Bytewax / Faust | 🟡 | `09/40/01` | Profundizar en C2 |
| Redis como online store | 🟡 | `10/25/03`, `10/42/02`, `09/27` | **Curso nuevo (C3)** |
| Triton | 🟡 | `10/29/03` (1 nota) | **Curso nuevo (C4)** |
| ONNX Runtime para modelos tabulares | ❌ | — | Dentro de C4 |
| TorchServe / vLLM / BentoML | ✅ | `09/30`, `06/20`, `09/42` | Solo comparativa en C4 |
| gRPC / WebSockets | ✅ | `10/31/07`, `13/02/03`, `10/30` | Reutilizar |
| SSE (streaming de tokens) | 🟡 | Disperso | **Curso nuevo (C5)** |
| Qdrant / pgvector | ✅ | `10/33`, `10/35`, `10/36` | Reutilizar |
| Grafana / Prometheus | 🟡 | `10/23/07` | **Curso nuevo (C6)** |
| Medición de latencia y load testing | ❌ | — | Dentro de C6 |
| OpenTelemetry / Langfuse | ✅ | `09/34`, `09/36` | Reutilizar |
| Decision models (Jev, Kev, Laya…) | ❌ | — | **Curso nuevo (C7)** |

---

## 2. Cursos nuevos

> La numeración se toma del siguiente número libre en cada módulo y se verifica antes de crear cada carpeta.
> Cada curso se cierra con un **puente** explícito hacia el proyecto que lo aplica.

### C1 · `10/46 - Apache Flink for Real-time ML` 🔴 Prioridad 1 → P1
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | Por qué Flink, mapa del curso, conexión con P1 |
| 01 | Stream Processing Model and Flink Architecture | Evento por evento vs micro-lotes, JobManager/TaskManager, slots, paralelismo, operator chaining |
| 02 | Time, Watermarks and Windows | Event time vs processing time, watermarks, eventos tardíos, ventanas tumbling/sliding/session |
| 03 | State, Checkpoints and Exactly-Once | Keyed state, RocksDB vs heap, checkpoints y savepoints, exactly-once de punta a punta con Kafka |
| 04 | Flink SQL for Real-time Features | Table API/SQL, ventanas de agregación (`pagos_10m`, `monto_10m`), conectores Kafka y upsert |
| 05 | PyFlink and the DataStream API | Cuándo bajar a DataStream, costo de las UDFs de Python, timers, process functions |
| 06 | Operating Flink on a Laptop | Backpressure, métricas, tuning de memoria en Docker con 8 GB, recuperación ante fallas |
| 07 | Capstone — Kafka → Flink → Redis Feature Pipeline | Mini pipeline que P1 extiende |

### C2 · `10/47 - Stream Processing Engines Compared` 🔴 Prioridad 1 → P1, P2, P3
Va **justo después de C1** para no perder el hilo.
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | Por qué comparar; cómo cada proyecto usa un motor distinto |
| 01 | Execution Models | Por evento (Flink) vs micro-lotes (Spark) vs continuous/Real-Time Mode de Spark (verificar estado actual) vs Python nativo |
| 02 | Spark Structured Streaming for ML Ingestion | `foreachBatch`, lotes de embeddings, triggers y estado. Complementa `10/27/04` sin duplicarlo |
| 03 | Python-Native Streaming: Bytewax, Faust, Quix + Kafka Streams | Dataflows en Python, costo operativo, cuándo bastan |
| 04 | Lab — Same Pipeline, Three Engines | Kafka → features en ventana → Redis en Flink, Spark y Quix Streams, midiendo el p95 de cada uno |
| 05 | Decision Framework and Interview Playbook | Árbol de decisión, trade-offs y cómo justificar la elección en system design |

### C3 · `09/43 - Real-time Feature Serving with Redis` 🟠 Prioridad 2 → P1, P2
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | Online vs offline store; dónde encaja Redis frente a Feast |
| 01 | Feature Data Modeling in Redis | Hashes vs JSON, diseño de claves, TTL, pipelining, Lua, memoria |
| 02 | Redis Streams vs Kafka | Cuándo cada uno; consumer groups |
| 03 | Point-in-Time Correctness and Train/Serve Skew | Carreras de datos, eventos enriquecidos vs lookup, consistencia |
| 04 | Latency Budgets for Online Serving | Presupuesto por tramo, p95/p99, conexiones y pools |

### C4 · `09/44 - High-Performance Model Serving - Triton and ONNX Runtime` 🟠 Prioridad 2 → P1
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | El mapa de serving: in-process vs servidor de inferencia |
| 01 | ONNX Runtime for Tabular Models | XGBoost/PyTorch → ONNX, sesiones, threads, benchmark contra nativo |
| 02 | Triton Deep Dive | Model repository, dynamic batching, instance groups, backends (ONNX, FIL para XGBoost, Python) |
| 03 | Triton Ensembles and perf_analyzer | Pipelines dentro de Triton, medición seria |
| 04 | Shadow Mode and Champion/Challenger | Cómo evaluar un modelo nuevo sin afectar las decisiones |
| 05 | Choosing a Serving Stack | Triton vs TorchServe vs vLLM vs BentoML vs in-process (enlaza los cursos existentes) |

### C5 · `10/48 - LLM Token Streaming - SSE, WebSockets and gRPC` 🟡 Prioridad 3 → P3
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | Por qué el streaming de tokens cambia la UX (TTFT) |
| 01 | Server-Sent Events Deep Dive | Protocolo, `Last-Event-ID`, reconexión, heartbeats |
| 02 | SSE in FastAPI and Behind Proxies | `StreamingResponse`/sse-starlette, buffering en nginx y Cloud Run, timeouts |
| 03 | Cancellation, Backpressure and Errors Mid-Stream | Cliente que se desconecta, cortar la generación, errores a mitad del stream |
| 04 | SSE vs WebSockets vs gRPC Streaming | Matriz de decisión (enlaza `10/30` y `10/31/07`) |

### C6 · `09/45 - Grafana and Latency Engineering for ML Systems` 🟠 Prioridad 2 → P1, P2, P3
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | De métricas sueltas a una historia de rendimiento |
| 01 | Prometheus Metrics Design for ML | Histogramas vs summaries, buckets, cardinalidad, percentiles correctos |
| 02 | Grafana Dashboards as Code | Provisioning, variables, paneles de throughput, lag y percentiles |
| 03 | SLOs, Alerting and Burn Rates | SLO de p95, alertas multi-ventana |
| 04 | Latency Measurement and Load Testing Methodology | p50/p95/p99, HdrHistogram, *coordinated omission*, carga por escalones, warm-up, cómo declarar el hardware |
| 05 | LLM Observability Dashboards | Tokens, costo, TTFT, tasa de aciertos de caché, OTel + Langfuse → Grafana |

### C7 · `06/34 - Decision Models - System One AI` 🟠 Prioridad 2 → P2
| # | Nota | Contenido clave |
|---|---|---|
| 00 | Welcome | System 1 vs System 2; tablas → XGBoost, decidir sobre texto → decision model, generar texto → LLM |
| 01 | How Decision Models Work | No autorregresivos, evaluación de opciones en paralelo, probabilidades, calibración (ECE) |
| 02 | Jev and the Landscape | Jev (TypeSafe) y **comparativa de todas las alternativas**: Kev, Laya, Von, NanoJev, SemIf, mini-jev, CLM, Decider, OpenDecision, jevlike, hosteadas (Solar Decide, pplx-decider, Clef…) y librerías (SetFit, Outlines, Instructor, DSPy) |
| 03 | Laya in Practice | Uso, fine-tune con HF, calibración por temperatura, inferencia en CPU |
| 04 | DIY: Classification Head on a Small Qwen | La técnica de cambiar la cabeza de lenguaje, para entender cómo funcionan por dentro |
| 05 | System 1 / System 2 Routing Patterns | Decision model + escalado a un agente LLM, umbrales de confianza, costo vs latencia |

> ⚠️ C7 trata un ecosistema de septiembre de 2026 que cambia rápido. Hay que verificar cada dato (tamaños, licencias, benchmarks) contra la fuente primaria al redactarlo.

### Resumen
| Curso | Notas | Prioridad | Alimenta a |
|---|---|---|---|
| C1 Flink | 8 | 🔴 1 | P1 |
| C2 Engines Compared | 6 | 🔴 1 | P1 · P2 · P3 |
| C3 Redis Feature Serving | 5 | 🟠 2 | P1 · P2 |
| C4 Triton + ONNX Runtime | 6 | 🟠 2 | P1 |
| C6 Grafana + Latency | 6 | 🟠 2 | P1 · P2 · P3 |
| C7 Decision Models | 6 | 🟠 2 | P2 |
| C5 Token Streaming | 5 | 🟡 3 | P3 |
| C8 GraphRAG & Knowledge Graphs | ~7 | 🟠 2 | P4 |
| **Total** | **~49 notas** | | |

### C8 · `06/35 - GraphRAG and Knowledge Graph Engineering` 🟠 Prioridad 2 → P4 (pendiente)
Neo4j y Cypher · modelado de grafos de conocimiento (incluido el grafo de un vault de Obsidian: wikilinks, tags, frontmatter) · GDS (Leiden, PageRank, proyecciones) · GraphRAG local vs global (Microsoft GraphRAG, LightRAG, neo4j-graphrag) · grafos temporales (Graphiti) · extracción y resolución de entidades (GLiNER) · Text2Cypher seguro · evaluación de GraphRAG. El vault solo tiene una nota (`06/13/04`) y menciones sueltas de Neo4j; Graphiti no aparece.

---

## 3. Los proyectos (resumen; detalle en el `PLAN.md` de cada repo)

| | P1 · Fraude en tiempo real | P2 · Router inteligente | P3 · RAG en vivo | P4 · Investigación de estudio (GraphRAG multiagente) |
|---|---|---|---|---|
| Modelo | **XGBoost** (ONNX) + challenger PyTorch en shadow | **Laya** + agente LangGraph para casos de baja confianza | **Haiku** (solo en el test final) + Qdrant/pgvector | **Gemma 4 31B** (texto y visión, free tier) vía P0 sobre el **vault de Obsidian** |
| Motor / núcleo | **Flink** (SQL) | **Quix Streams** | **Spark Structured Streaming** | **Neo4j + LangGraph** (GraphRAG) |
| Transporte | Kafka (KRaft) | Redpanda (API Kafka) + FastAPI | Redpanda + outbox CDC + **SSE** | MCP + SSE |
| Cloud | Local | Local | **GCP** (Cloud Run + Qdrant Cloud free + Vertex AI Pipelines) | Local |
| Gasto | $0 | $0 | $0 hasta el test final | $0 |
| Absorbe del CV | Evidently, backtesting en contenedor sobre MinIO, auditoría de explicaciones | `judgekit` (Evaluation Suite completa) + Langfuse | Auditoría de alucinaciones, Vertex AI Pipelines, caché invalidada por CDC | Multi-Agent Research System completo + GraphRAG sobre las notas de estudio |
| Frase del CV | "Kafka + Flink, X ev/s, p95 de Y ms" | "Router System 1/2: X% resuelto en Y ms, Z% menos costo de LLM" | "RAG con índice en vivo: N s de frescura, TTFT de X ms, faithfulness de Y" | "Agentes de investigación de estudio sobre ~1.000 notas de Obsidian: X% en multi-hop, +Y pp sobre RAG vectorial, Z notas desactualizadas detectadas" |

**P0 · `llm-gateway`** (base de todos): absorbe por completo el *LLM Edge Gateway* en Go del CV (caché semántica obligatoria, circuit breaker obligatorio, fallback Gemma 3 1B, Groq, UI de chat) y agrega Gemma 4 vía Google AI Studio, hedging, concurrencia adaptativa, cuotas de free tiers, guardrails y presupuesto atómico.

### Mapa de integración
```
                ┌───────────────────────────────┐
                │  P0 · llm-gateway (Python)     │  ← nuevo, desde cero (reemplaza al de Go)
                │  cache semántica · breaker ·   │     punto único de control de costos
                │  mock / Ollama / Haiku         │
                └──────▲────────▲─────────▲──────┘
                       │        │         │
   P1 explicaciones ───┘        │         └─── P3 generación (Haiku en el test final)
   (async, LLM local)           │
                       P2 agente System 2 + juez (LLM local)

   P2 Router ──(intención "pregunta de conocimiento")──► P3 RAG
   P2 Router ──(intención "disputa de pago")───────────► consulta de decisiones de P1
   Vault de Obsidian (Learning) ──(lectura)──► grafo de P4 ──(notas de investigación)──► carpeta del vault
   P2 judgekit ──(librería)──► P1 · P3 · P4          P3 kb-changes ──(invalidación por tags)──► caché de P0
   P4 agentes ──(smart · vision · extract · judge)──► P0
```
Cada proyecto funciona **solo**. La integración es un extra demostrable al final.

### Skills del CV integradas
Ver la matriz completa P0–P4 en **§8.4** (y la trazabilidad de los proyectos del CV en §8.1–8.3).

---

## 4. Orden de ejecución

Los cursos y los proyectos se intercalan: cada curso se escribe justo antes de usarse.

| Fase | Trabajo | Sale |
|---|---|---|
| **F0** ✅ | Planes de los proyectos (P0 gateway · P1 · P2 · P3) + **setup M0 de los 4 repos** (git, CI, tests, Compose, `doctor`) | `AI Engineer proyects/` |
| **F2.5** | **Implementar P0 `llm-gateway`** M1–M2 (antes del M7 de P1 y del M5 de P2); M3–M7 antes de la prueba final de P3 | Gateway usable por los 3 |
| **F1** ✅ | C1 Flink → C2 Engines Compared (`10/46`, `10/47`, commit cc64557) | 14 notas |
| **F2** ✅ | C3 Redis → C4 Triton/ONNX → C6 Grafana/Latency (`09/43`, `09/44`, `09/45`) | 17 notas |
| **F3** | **Implementar P1** | Repo P1 + README + resultados |
| **F4** ✅ | C7 Decision Models (`06/34`) | 6 notas |
| **F5** | **Implementar P2** | Repo P2 + README |
| **F6** ✅ | C5 Token Streaming (`10/48`) | 5 notas |
| **F7** | **Implementar P3** (local → GCP solo en el test final) | Repo P3 + README |
| **F7.5** | C8 GraphRAG → **Implementar P4** (M0 ✅ hecho el 2026-10-06) | Repo P4 + README |
| **F8** | Integración con el gateway + pruebas finales (el único momento con gasto) | Cifras finales del CV |
| **F9** | `⏳ 16GB`: re-medir P1 con carga máxima, CDC completo en P3, Langfuse self-hosted | Cifras actualizadas |
| **F10** | Cierre: actualizar el índice maestro, el Continuity Prompt y el Skills Tree, y **borrar este archivo** | — |

---

## 5. Mantenimiento del vault (al cerrar cada curso)
- [ ] Agregar el curso a `00 - Indice Maestro de Cursos.md`
- [ ] Actualizar `Continuity Prompt.md` (estado, conteo de notas, gaps cerrados)
- [ ] Actualizar `Skills Tree - The AI-ML Engineer Growth Map.md`
- [ ] Enlaces cruzados: nota existente ↔ curso nuevo (Kafka `10/29/01`, Spark `10/27/04`, Triton `10/29/03`, OTel `09/34`…)
- [ ] Nombres de archivo sin `:` ni caracteres inválidos en Windows; rutas cortas (por el límite de longitud)
- [ ] Commit por curso: `feat: add <course> (N notes)`

---

## 6. Pendientes `⏳ 16GB`
| Pendiente | Proyecto |
|---|---|
| Prueba de carga máxima con 20k ev/s, Grafana activa, 2 TaskManagers, 3 brokers | P1 |
| Pruebas de falla con réplicas reales | P1 |
| Debezium + Kafka Connect en lugar del CDC ligero | P3 |
| Langfuse self-hosted (Postgres + ClickHouse) junto con los stacks | P2 · P3 · P4 |
| Microsoft GraphRAG sobre el corpus completo | P4 |
| LLM local de 4B cuantizado (en lugar de 1-2B) | P2 · P3 |

---

## 7. Registro de decisiones
| Fecha | Decisión |
|---|---|
| 2026-10-05 | Flink en un curso propio + un curso comparativo aparte, numerado justo después |
| 2026-10-05 | Un motor por proyecto: Flink (P1), Bytewax (P2), Spark SS (P3) |
| 2026-10-05 | Un modelo por proyecto: XGBoost (P1), Laya (P2), Haiku (P3) |
| 2026-10-05 | C7 documenta la comparativa de todos los decision models; los proyectos usan solo Laya |
| 2026-10-05 | Solo P3 en GCP; costo mínimo; Terraform para crear y destruir |
| 2026-10-05 | ~~Todas las llamadas al LLM pasan por el LLM Edge Gateway (Go) existente~~ → reemplazado el 2026-10-06 |
| 2026-10-05 | Diseño base para 8 GB de RAM y 4 GB de VRAM; lo más pesado queda como `⏳ 16GB` |
| 2026-10-05 | Cada proyecto tiene un README progresivo: teoría y visión macro primero, detalle técnico al final de cada componente |
| 2026-10-05 | F1 completada. Hallazgo: Bytewax sin release desde nov-2024 (v0.21.1); Quix Streams activo (v3.27.0, sep-2026) → **propuesto** cambiar el motor de P2 a Quix Streams (pendiente de confirmación del usuario) |
| 2026-10-06 | F2 completada. Hallazgo: **TorchServe archivado** (ago-2025) → el curso `09/30` necesita aviso de deprecación (pendiente de confirmación); P1 sigue sin depender de él |
| 2026-10-06 | **Confirmado por el usuario:** P2 usa **Quix Streams** en lugar de Bytewax (plan de P2 actualizado). Aviso de deprecación agregado a `09/30 - TorchServe` |
| 2026-10-06 | F4 y F6 completadas (adelantadas a F3/F5 por decisión del usuario). Hallazgos: Laya colapsa con >20 opciones (P2 ya usa ~12 rutas), multilingual sin calibrar, latencia CPU incierta (riesgos R1b–R1d en P2); FastAPI trae SSE nativo con ping cada 15 s |
| 2026-10-06 | **Gateway nuevo:** el usuario descarta el gateway en Go del CV. Se crea **P0 `llm-gateway`** (Python, FastAPI, desde cero): API compatible con OpenAI, fallback, circuit breakers, presupuesto atómico en Redis, caché exacta y semántica, SSE con cancelación. Los 4 repos tienen su setup M0 en `Documents/AI Engineer proyects/` (20 tests en verde, ruff limpio). Faltan **Docker Desktop y uv** en la máquina |
| 2026-10-06 | Los 4 repos quedan **públicos** con CI en verde; commit `edbf348` de Learning subido. Se mantienen 4 repos + P4 (el gateway no se fusiona) |
| 2026-10-06 | **v2 de los planes:** los 3 proyectos del CV quedan aparte y se absorben en P0–P4 (§8). P0 v2 (todo el gateway en Go + caché semántica y breaker obligatorios, Groq, Gemma 4, hedging, guardrails, UI). `judgekit` nace en P2 (Evaluation Suite + Langfuse). P1 suma Evidently, SageMaker Processing local y el grafo para P4. P3 suma auditoría de alucinaciones, Vertex AI Pipelines y caché invalidada por CDC. **P4 `graphrag-research-agents`** creado (plan + M0 + repo público). Nuevo curso pendiente C8 GraphRAG |
| 2026-10-06 | **v3:** (1) **sin SageMaker**: la evaluación batch corre en contenedores propios con KFP local sobre MinIO; (2) **P4 se re-enfoca** en investigación de estudio sobre el **vault de Obsidian** (`Learning/SW-ML-AI Engineering`, ~1.000 notas, ~4.700 wikilinks): grafo determinista desde wikilinks y tags, detección de notas desactualizadas, notas de investigación escritas en una carpeta propia del vault; (3) **Tavily reemplazado** por SearXNG self-hosted + Crawl4AI + arXiv/OpenAlex (Exa opcional); (4) se elimina la integración P1 → P4 (`entity-edges`) |


---

## 8. Trazabilidad: proyectos del CV → P0–P4 (sin gaps)

> **Decisión del 2026-10-06:** los tres proyectos actuales del CV quedan **aparte** (archivados como antecedentes). Todo lo que demuestran se implementa en P0–P4 como **componentes reales**, no como ideas. Cuando P0–P4 tengan resultados, el CV los reemplaza.

### 8.1 High-Performance LLM Edge Gateway (Go) → **P0**
| Tecnología, método o componente | Dónde | Mejora |
|---|---|---|
| Caché semántica (embeddings + KNN en Redis, HNSW COSINE, umbral, TTL, contador de hits, write-back asíncrono) | P0 §6 | Obligatoria y en capas: namespace, umbral calibrado, guardas léxicas, verificador, invalidación por tags (P3) |
| `nomic-embed-text` vía Ollama | P0 §6.6 | Baseline medido frente a e5 ONNX |
| Circuit breaker de 3 estados | P0 §4.3 | Tres disparadores: consecutivas, tasa de error y llamadas lentas |
| Fallback local Gemma 3 1B (CPU) | P0 §2 | Último eslabón de toda cadena |
| Groq (OpenAI-compatible, free tier) | P0 §3 | `OpenAICompatProvider` genérico + cuotas salientes |
| Streaming SSE OpenAI-compatible, `X-Accel-Buffering` | P0 §7 | Cancelación, stalls, regresión del bug de contexto |
| Auth Bearer en tiempo constante, fail-closed | P0 §8.3 | Keys por cliente con hash |
| Headers `X-Cache-Status`, `X-Provider`, `X-Cache-Similarity`, `X-Latency-Ms` | P0 ADR-14 | Ampliados |
| UI de chat integrada en `GET /` | P0 §9.1 | Selector de alias, breakers, presupuesto |
| `/health`, `/stats`, apagado ordenado, config 12-factor, ejemplos de clientes | P0 §8–9 | Prometheus/OTel, drain con liquidación |
| Cifras del CV (34–90 ms, 30–40%, 99.9%, 10K req/s) | P0 §0.3 | Re-medidas con método (G9–G13) |

### 8.2 Enterprise MLOps: Automated LLM Evaluation Suite → **`judgekit` (P2) + P1 + P3**
| Tecnología, método o componente | Dónde |
|---|---|
| Framework asíncrono de LLM-as-a-Judge (asyncio, aiohttp, Pandas) | P2 §12 (`judgekit`) |
| Gemma 4 31B como *golden evaluator* | P0 alias `judge_golden` → P2 §12, P3 §14 |
| Hugging Face Transformers, Tokenizers y Evaluate; token matching | `judgekit` (P2 §12) |
| Auditoría de alucinaciones | P2 (rationale del agente), P1 (explicaciones), P3 (NLI + panel de jueces), P4 (Critic) |
| Benchmarking de alineamiento, seguridad y marca; gating de producción | `judgekit gate` (P2 §12) |
| Jobs de evaluación distribuida sobre S3 versionado | P2 §12 (evaluación) y P1 §13 (backtesting): contenedores propios con KFP local + MinIO (**sin SageMaker**) |
| Vertex AI Pipelines (reentrenar o re-promptear al degradarse) | P2 §12 (KFP local) → P3 §14 (ejecución en Vertex en la prueba final) |
| Monitoreo de modelos (drift semántico) | Evidently en P1 §13 y P2 §12 |
| Gobernanza de datos | P1 §13 y P2 §12 (dataset cards, manifiestos, linaje, Presidio) |
| **Langfuse** (mejora pedida) | P2 §12.2 (datasets, experiments, scores, LLM-as-a-judge gestionado, prompts, anotación), P3 y P4 |

### 8.3 Hybrid RAG & Multi-Agent Research System → **P4** (+ P3)
| Tecnología, método o componente | Dónde |
|---|---|
| LangGraph cíclico; agentes Research, Fact-Auditing, Synthesis; Leader determinista | P4 §5 |
| Qdrant + ChromaDB tras `VectorStore`; `bge-small-en-v1.5`; `RecursiveCharacterTextSplitter` 512/64; umbral 0.70 | P4 §3–4 y P3 §14 (baselines) |
| Búsqueda web con backoff y modo solo-RAG (Tavily **reemplazado**: el usuario lo probó y rinde mal) | P4 §6: SearXNG self-hosted + Crawl4AI, arXiv y OpenAlex para fuentes académicas, Exa opcional; todo detrás de `web-mcp` |
| Gemma 4 multimodal y *function calling*; auditoría visual | P4 §5.3, P0 alias `vision` |
| Ejecución de código para verificación | P4 §6.3 (sandbox vía MCP) |
| Compactación de contexto (ventana 5 + resumen cada 10) | P4 §7.2 |
| MLflow file-based con `metrics.jsonl`, saneamiento y hashes | P4 §10, P3 §14 |
| Abstracción de proveedores, fábrica, `mock`, fail-closed, "evidencia insuficiente" | P4 §5.5, requisitos EARS |
| **Harness SDD** (EARS, compuertas humanas, contratos de rol, ADRs JSON, contexto < 20%, diffs < 400) | P4 §9 (completo); EARS también en P0 §11 |
| **GraphRAG** (mejora pedida) | P4 §4: el vault de Obsidian como grafo (wikilinks deterministas + conceptos extraídos), Neo4j, GDS Leiden, neo4j-graphrag, Graphiti, GLiNER, Text2Cypher; baselines Microsoft GraphRAG y LightRAG |

### 8.4 Skills del CV y dónde quedan demostradas
| Skill del CV | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| Python · FastAPI | ✅ | ✅ | ✅ | ✅ (SSE) | ✅ (SSE) |
| PyTorch | — | Challenger | Fine-tune de Laya | — | — |
| ONNX Runtime | Embeddings y guardrails | Scorer | Laya int8 | Embeddings + NLI | Embeddings |
| MLflow | — | Registry | Evals | Evals | Evals (saneado) |
| PostgreSQL | — | Auditoría | Decisiones | Fuente CDC + pgvector | Checkpointer |
| Redis | Budget, caché semántica, cuotas | Online store | Colas | Vía P0 | Caché web |
| Qdrant | (plan B) | — | — | ✅ | ✅ |
| Hugging Face | Tokenizers | — | ✅ | Embeddings | GLiNER, embeddings |
| LangChain / LangGraph | Clientes | — | Agente System 2 | RAG agéntico | Multiagente + MCP |
| Langfuse | Exportador | — | ✅ (columna vertebral) | ✅ | ✅ |
| LLMOps / LLM-as-a-Judge | — | Auditoría de explicaciones | `judgekit` | Panel de jueces | Critic |
| Harness Engineering | EARS | — | Eval harness | Eval harness | SDD completo |
| Docker | ✅ | ✅ | ✅ | ✅ | ✅ |
| GCP | — | — | (KFP compilado) | Cloud Run + Vertex AI Pipelines | — |
