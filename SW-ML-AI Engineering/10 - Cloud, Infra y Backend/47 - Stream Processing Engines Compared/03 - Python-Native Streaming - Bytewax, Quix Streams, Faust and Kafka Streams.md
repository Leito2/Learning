# 🐍 03 - Python-Native Streaming — Bytewax, Quix Streams, Faust and Kafka Streams

Your model is a Python object already loaded in memory, your feature logic is NumPy, and your team writes Python all day. Shipping every event to a JVM cluster that then calls back into Python — or into a model server over the network — adds a hop, a deployment, and a language boundary to every change. Python-native streaming engines remove that boundary. The price is CPU per record and, sometimes, project maturity. This note shows both sides with numbers.

## 🎯 Learning Objectives
- Explain the architecture of **Bytewax** (Python on a Rust dataflow), **Quix Streams** (Kafka-native library with RocksDB state), and **Faust** (asyncio agents)
- Understand **Kafka Streams** as the reference model for library-style stream processing
- Estimate **Python throughput per process** and the number of processes needed for a target rate
- Compare state, windows, delivery guarantees, scaling, and **maintenance signals** across engines
- Implement the same keyed windowed feature in Quix Streams and Bytewax
- Choose a Python engine for P2 with evidence, including project health

## Introduction

There are two broad ways to put Python into a streaming pipeline. The first wraps Python inside a JVM engine (PyFlink, PySpark): the engine owns scheduling, state, and fault tolerance, and calls Python for user logic, paying serialization at the boundary ([[../46 - Apache Flink for Real-time ML/05 - PyFlink and the DataStream API|PyFlink]]). The second makes Python the host: the engine is a Python library (or a Rust core with a Python API), runs inside your process, and uses Kafka itself for coordination and state backup.

The second family is attractive for ML teams because the streaming job *is* an ordinary Python service: same dependencies as training, same tests, same container image, the model loaded once per process. That is exactly P2's situation — a support-message router where every event goes through a decision model (Laya) running in-process on CPU.

The third option, **Kafka Streams**, is not Python at all, but it defines the library-style model that Faust and Quix Streams follow: stream processing as a library, partitions as the unit of parallelism, local state backed by Kafka changelog topics. Understanding it makes the Python engines easy to reason about.

---

## 1. The Problem and Why This Solution Exists

### The cost of crossing languages

If a JVM engine computes features and a Python service runs the model, each event crosses at least one network hop and two serializations. If the JVM engine calls Python in-process (PyFlink, PySpark UDFs), each record still crosses a language boundary. For heavy ML logic per event — tokenization plus an encoder forward pass — that boundary is small compared to the model; for light logic, it dominates.

### The cost of staying in Python

CPython executes one thread of Python bytecode at a time per process (the GIL). A Python streaming process therefore has a per-record CPU cost $c$ and a capacity:

$$
\mu_{\text{proc}} = \frac{1}{c}
\qquad
n_{\text{proc}} = \left\lceil \frac{\lambda}{\rho_{\text{target}} \cdot \mu_{\text{proc}}} \right\rceil
\le N_{\text{partitions}}
$$

With $c = 120\,\mu s$ (JSON parse + feature update + produce), one process handles ~8,300 ev/s at 100% CPU; at a target utilization $\rho = 0.6$ (to protect tail latency, see [[01 - Execution Models|note 01]]), 5,000 ev/s needs one process, 20,000 ev/s needs four — **and four Kafka partitions at least**. If the per-record cost includes an in-process model call amortized over a batch of $k$ events, $c = c_{\text{parse}} + c_{\text{model}}/k$, and batching becomes the main throughput lever.

💡 **Tip:** free-threaded CPython builds (PEP 703, experimental from 3.13) may eventually relax the GIL limit, but library support for streaming engines is not something to assume. Plan capacity by **processes × partitions**.

---

## 2. Conceptual Deep Dive

### 2.1 Kafka Streams: the reference model

- A **topology** of processors (`KStream` for event streams, `KTable` for changelog-backed tables) runs inside each application instance.
- **Tasks** map 1:1 to input partitions; Kafka's consumer-group protocol assigns tasks to instances. Adding instances scales out (up to the partition count).
- **State stores** are local RocksDB instances per task, each backed by a compacted **changelog topic**. After a rebalance, the new owner restores the store by replaying the changelog; **standby replicas** keep warm copies to shorten that.
- Re-keying (`groupBy` with a new key) writes to an internal **repartition topic** — a broker round-trip instead of an in-memory shuffle.
- `processing.guarantee=exactly_once_v2` wraps consume–process–produce and state changelog writes in Kafka transactions per commit interval.
- **Interactive queries** expose local state over RPC — the app can serve its own features.

### 2.2 Quix Streams: Kafka Streams ideas, pure Python

Quix Streams (v2 was a full rewrite; v3.x in 2026) offers a **StreamingDataFrame** API over Kafka topics:

- `Application` manages consumer, producer, commits, and state.
- `sdf.apply/filter/update`, `group_by` (re-keying through a repartition topic), and windows: `tumbling_window`, `hopping_window`, `sliding_window`, with `.count()/.sum()/.agg(...)` and emission via `.final()` (on window close) or `.current()` (on every update).
- **State** per key in local RocksDB, backed by changelog topics — the Kafka Streams recovery model.
- `processing_guarantee="exactly-once"` uses Kafka transactions; default is at-least-once.
- Scaling: run more processes in the same consumer group.

`.current()` is the Python analogue of an `OVER` window for per-event features: every incoming event emits the window's updated aggregate, including itself.

### 2.3 Bytewax: Python API, Rust dataflow runtime

- A `Dataflow` is built with operators: `op.input`, `op.map`, `op.key_on`, `op.stateful_map`, windowing operators (`bytewax.operators.windowing`: tumbling, sliding, session), `op.output`.
- The runtime (based on Timely Dataflow) runs **workers** (threads/processes, `python -m bytewax.run module:flow -w 2`), exchanging keyed data between workers.
- **Recovery** snapshots operator state per epoch into local recovery partitions (SQLite files); on restart it resumes from the last snapshot and the matching input offsets.
- Python callbacks run per record — throughput is bounded by the Python cost per operator.

### 2.4 Faust (faust-streaming)

Faust (originally from Robinhood, now maintained as the community fork `faust-streaming`) expresses stream processing as **asyncio agents** that iterate over topics, with **tables** (RocksDB + changelog) and windowed tables. It fits teams already built on asyncio; its feature set and documentation have evolved more slowly than Quix Streams'.

### 2.5 Side-by-side

| | Kafka Streams | Quix Streams | Bytewax | Faust |
|---|---|---|---|---|
| Language | Java/Kotlin | Python | Python API, Rust runtime | Python (asyncio) |
| Deployment | Library in your service | Library in your service | Python process(es) running a dataflow | Library in your service |
| Parallelism | Partitions → tasks → instances | Partitions → processes | Workers (threads/processes) | Partitions → workers |
| State | RocksDB + changelog | RocksDB + changelog | In-memory + recovery snapshots | RocksDB tables + changelog |
| Re-keying | Repartition topic | Repartition topic | In-runtime exchange | Repartition topic |
| Windows | Tumbling, hopping, sliding, session | Tumbling, hopping, sliding | Tumbling, sliding, session | Tumbling, hopping (tables) |
| Exactly-once | `exactly_once_v2` | Kafka transactions (opt-in) | At-least-once + recovery | At-least-once typical |
| External cluster | None (Kafka only) | None | None | None |

### 2.6 Maintenance signals are an engineering input

Choosing a dependency for a portfolio or production system includes its **health**: release cadence, open-issue response, number of active maintainers. A snapshot from the GitHub API (October 2026):

| Project | Latest release | Last push | Stars |
|---|---|---|---|
| quixio/quix-streams | **v3.27.0 — 2026-09-30** | 2026-10-01 | ~1.6k |
| faust-streaming/faust | v0.15.3 — 2026-08-23 | 2026-09-28 | ~1.9k |
| bytewax/bytewax | **v0.21.1 — 2024-11-25** | 2026-06-20 | ~2.1k |

⚠️ **Warning:** Bytewax has had **no tagged release for almost two years** at the time of writing, although the repository still receives occasional commits. That does not make it unusable, but it is a risk signal: Python-version support, dependency updates, and bug fixes may lag. Re-check before committing a project to it, and prefer an engine with an active release cadence when the choice is otherwise close.

---

## 3. Production Reality

### Rebalances and state restoration

Library engines restore state by **replaying changelog topics** when partitions move. A process restart or a scale-out triggers a rebalance; while state restores, those partitions do not process events. With a changelog of $S$ bytes and restore throughput $r$:

$$
T_{\text{restore}} \approx \frac{S}{r}
$$

Keep state small (TTL, windows instead of unbounded tables), keep changelogs compacted, and in Kafka Streams use standby replicas for faster failover.

### Python service hygiene

- **One model per process**, loaded once at startup; never per message.
- **Micro-batch model calls** where possible (collect $k$ messages or wait $\tau$ ms), exactly like P2's Laya calls.
- **Partitions ≥ planned processes** — decide partition count up front; increasing it later reshuffles keys.
- Expose Prometheus metrics from the process (per-stage latency, batch size, lag).

Caso real: ML teams frequently replace a "Flink computes features → HTTP call to a Python model server" pipeline with a single Python stream processor that loads the model in-process, because the network hop and the second deployment cost more latency and engineering time than the Python CPU overhead. They keep Flink for the few high-throughput, millisecond-sensitive features that Python cannot sustain.

Caso real: for P2 (support-message router), the per-message work is dominated by an encoder forward pass (tens of milliseconds on CPU, amortized by batching), not by the framework. Framework overhead of ~100 µs per message is noise next to the model; language fit, in-process models, and project health decide.

### Recommendation for P2 (evidence-based)

| Criterion | Quix Streams | Bytewax |
|---|---|---|
| Per-event emission (`current()`) for routing context | ✅ | ✅ (stateful_map) |
| Kafka-native state recovery | ✅ changelog topics | Snapshots to local files |
| Exactly-once option | ✅ | — |
| Release cadence (Oct 2026) | ✅ active (monthly) | ⚠️ none since Nov 2024 |
| Fit with Redpanda (Kafka API) | ✅ | ✅ |

On current evidence, **Quix Streams is the lower-risk choice for P2**; Bytewax remains a valid alternative to demonstrate the Rust-dataflow model. The decision belongs to the project plan and should be revisited when it is implemented.

---

## 4. Code in Practice

### The same feature in Quix Streams

```python
# quix_features.py — per-user 10-min count emitted on every message (OVER-like), plus surge counts
from datetime import timedelta
from quixstreams import Application

app = Application(broker_address="localhost:9092", consumer_group="p2-features",
                  auto_offset_reset="latest", processing_guarantee="at-least-once")
messages = app.topic("inbound-messages", value_deserializer="json")
enriched = app.topic("messages-enriched", value_serializer="json")

sdf = app.dataframe(messages)
sdf = sdf.group_by("customer_id")                                # re-key via a repartition topic
cnt = (sdf.sliding_window(duration_ms=timedelta(minutes=10)).count().current())  # emit per event
cnt = cnt.apply(lambda w: {"customer_id_msgs_10m": w["value"], "window_end": w["end"]})
cnt.to_topic(enriched)

if __name__ == "__main__":
    app.run()          # scale out: start more processes with the same consumer_group
```

### The same feature in Bytewax

```python
# bytewax_features.py — stateful_map keeps a per-customer deque; emits on every message
import json
from collections import deque
from datetime import timedelta
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.connectors.kafka import KafkaSource, KafkaSink, KafkaSinkMessage

def update(state, msg):
    q = state or deque()
    t = msg["ts"]
    q.append(t)
    while q and q[0] < t - 600:
        q.popleft()
    return q, {**msg, "customer_msgs_10m": len(q)}

flow = Dataflow("p2-features")
raw = op.input("in", flow, KafkaSource(["localhost:9092"], ["inbound-messages"]))
msgs = op.map("parse", raw, lambda m: json.loads(m.value))
keyed = op.key_on("by_customer", msgs, lambda m: m["customer_id"])
feats = op.stateful_map("velocity", keyed, update)
out = op.map("encode", feats, lambda kv: KafkaSinkMessage(kv[0], json.dumps(kv[1])))
op.output("out", out, KafkaSink(["localhost:9092"], "messages-enriched"))
# run: python -m bytewax.run bytewax_features:flow -w 2   (recovery flags configure snapshots)
```

⚠️ **Warning:** operator and connector signatures have changed across Bytewax 0.1x releases, and Quix Streams 3.x renamed several window helpers over time. Treat the snippets as shapes and check the docs of the version you pin.

### ❌/✅ Loading the model

```python
# ❌ Model loaded per message: seconds of latency per event, memory churn
def route(msg):
    model = load_onnx("laya-int8.onnx")
    return model.predict(msg["text"])

# ✅ Loaded once per process; batched calls amortize the forward pass
MODEL = load_onnx("laya-int8.onnx")
def route_batch(msgs):
    return MODEL.predict([m["text"] for m in msgs])
```

### 📦 Compression code: how many Python processes do I need?

```python
# 📦 Compression code: measure per-record Python cost and size the process count
# Covers: c per record, μ per process, n processes at a target utilization, partition floor
import json, math, time
from collections import defaultdict, deque

state = defaultdict(deque)
msg = json.dumps({"customer_id": "c_42", "ts": 0.0, "text": "me cobraron dos veces"})

def handle(raw: str, t: float):
    m = json.loads(raw)                   # parse
    q = state[m["customer_id"]]
    q.append(t)
    while q and q[0] < t - 600:
        q.popleft()
    return json.dumps({**m, "msgs_10m": len(q)})   # serialize result

N = 100_000
start = time.perf_counter()
for i in range(N):
    handle(msg, i * 0.001)
c_logic = (time.perf_counter() - start) / N
FRAMEWORK = 80e-6                         # assumed consume/produce/commit overhead per record
MODEL = 15e-3 / 16                        # 15 ms encoder forward pass, batched by 16 messages
for label, c in (("logic only", c_logic), ("+ framework", c_logic + FRAMEWORK),
                 ("+ model (P2)", c_logic + FRAMEWORK + MODEL)):
    mu = 1 / c
    plan = ", ".join(f"{lam:,}/s→{math.ceil(lam / (0.6 * mu))}p" for lam in (2_000, 5_000, 20_000))
    print(f"{label:<13} c≈{c*1e6:7.1f} µs  μ≈{mu:>9,.0f} ev/s/process  processes@ρ=0.6: {plan}")
# ¡Sorpresa! The pure logic is microseconds; the batched model call dominates and sets the process count.
```

---

## 🎯 Key Takeaways
- Python-native engines run **inside your Python service**: same language, same model in memory, no JVM boundary.
- Throughput scales by **processes × partitions**; size with $n = \lceil \lambda / (\rho\,\mu_{\text{proc}}) \rceil$.
- **Kafka Streams** defines the library model: tasks per partition, RocksDB state with changelog topics, repartition topics.
- **Quix Streams** brings that model to Python with windows, `current()` per-event emission, and opt-in exactly-once.
- **Bytewax** pairs a Python API with a Rust dataflow runtime and snapshot recovery — but has had **no release since Nov 2024**.
- Project **health** (release cadence, maintainers) is a legitimate engineering criterion — check it before committing.
- For model-heavy events, the model cost dominates; batch model calls and load models once per process.

## References
- Kafka Streams documentation — architecture, state stores, processing guarantees, interactive queries
- Quix Streams documentation (v3.x) — StreamingDataFrame, windowing, state, processing guarantees
- Bytewax documentation — operators, windowing, recovery, `bytewax.run`
- faust-streaming documentation — agents, tables, windowing
- PEP 703 — Making the Global Interpreter Lock Optional in CPython
- GitHub API snapshot of repository releases (accessed 2026-10-05)
- [[../../09 - MLOps y Produccion/40 - Real-time ML Systems/01 - Streaming Feature Engineering - Kafka, Faust-Bytewax, and Online Aggregations|Streaming Feature Engineering (Bytewax basics)]]
- [[02 - Spark Structured Streaming for ML Ingestion|Previous: Spark for ML Ingestion]] · [[04 - Lab - Same Pipeline Three Engines|Next: Lab]]
