# 🔧 06 - Operating Flink on a Laptop

Flink's defaults are tuned for throughput on clusters with dozens of cores and gigabytes per TaskManager. On a 4-core laptop with 8 GB of RAM shared with Kafka, Redis, a load generator, and Windows itself, those defaults give you a ~100 ms latency floor, a TaskManager that does not start, or a job that silently backpressures. This note is the operations manual: where the memory goes, which knobs move p95, and how to read what Flink is telling you.

## 🎯 Learning Objectives
- Break down the **TaskManager memory model** and size it for a 1.2 GB budget
- Tune the latency knobs: `execution.buffer-timeout`, chaining, watermark interval, Kafka client settings
- Read **backpressure** metrics (`busyTime`, `backPressuredTime`, buffer pool usage) and locate the bottleneck
- Export Flink metrics to **Prometheus** and know which ones matter for an SLO
- Design a **parameter sweep** (experiment E3) that proves the effect of a knob
- Avoid laptop-specific traps: CPU oversubscription, thermal throttling, WSL2 memory limits

## Introduction

Notes 01–05 explained *what* Flink does. This one is about *running it well* on modest hardware — which is also the best way to understand it on large hardware. Every production Flink problem eventually reduces to one of three questions: is there enough memory of the right kind, where is time being spent, and which operator is holding everyone else back?

On a laptop those questions are sharper because there is no slack. The P1 fraud pipeline budgets roughly **650 MB for the JobManager and 1.25 GB for the TaskManager** inside a 5 GB WSL2 VM, with about two CPU threads for Flink. Within that envelope the goal is a Flink segment that contributes a few milliseconds — not a hundred — to an 80 ms end-to-end p95.

---

## 1. The Problem and Why This Solution Exists

### Defaults optimize for throughput

Batching is how distributed systems get throughput: bigger network buffers, longer timeouts, mini-batches, bundled Python calls. Each of these adds latency per record. Flink exposes the knobs precisely because there is no universal right answer — a nightly backfill and a fraud scorer want opposite settings.

### Memory is not one number

A JVM process uses heap, off-heap direct memory, metaspace, and native memory (RocksDB). Flink divides `taskmanager.memory.process.size` into named pools so that RocksDB, network buffers, and user code cannot starve each other. Setting only the container limit — or only `-Xmx` — leads to either `OOMKilled` containers or `OutOfMemoryError: Direct buffer memory`.

---

## 2. Conceptual Deep Dive

### 2.1 The TaskManager memory model

$$
\underbrace{M_{\text{process}}}_{\texttt{process.size}}
= \underbrace{M_{\text{heap}} + M_{\text{managed}} + M_{\text{network}} + M_{\text{off-heap}}}_{\text{total Flink memory}}
+ M_{\text{metaspace}} + M_{\text{overhead}}
$$

| Pool | What uses it | Default sizing rule (verify per version) |
|---|---|---|
| Framework + task **heap** | Flink internals, your operators' objects, heap state backend | Remainder of total Flink memory |
| **Managed** memory | RocksDB block cache & write buffers, Python workers, batch sorting | Fraction ≈ 0.4 of total Flink memory |
| **Network** memory | Network buffers for shuffles | Fraction ≈ 0.1, clamped to [64 MB, 1 GB] |
| Framework/task off-heap | Direct memory used by Flink/user code | Small fixed amounts |
| JVM **metaspace** | Loaded classes | ≈ 256 MB |
| JVM **overhead** | Thread stacks, GC, native code | Fraction ≈ 0.1, clamped to [192 MB, 1 GB] |

For `process.size = 1200m`: overhead ≈ 192 MB (the minimum), metaspace ≈ 256 MB, leaving ≈ 750 MB of total Flink memory → ≈ 300 MB managed (RocksDB), ≈ 75 MB network, and the rest heap. That is enough for P1's state (tens of MB) — but note how **metaspace and overhead eat 37% of a small TaskManager**. One TaskManager with 4 slots is far more efficient than four small TaskManagers.

⚠️ **Warning:** Flink validates these pools at startup. If explicit settings contradict each other (e.g., a fixed managed size larger than what remains), the TaskManager refuses to start with an `IllegalConfigurationException` — read the message; it prints the computed breakdown.

💡 **Tip:** the TaskManager log prints the final memory configuration at startup. Copy it into your project notes the first time — it is the ground truth for your budget table.

### 2.2 The latency knobs

| Knob | Default (verify) | Latency-oriented value | Effect |
|---|---|---|---|
| `execution.buffer-timeout` | 100 ms | **1–5 ms** | Max time a record waits in a non-full network buffer. `0` flushes after every record (highest CPU) |
| Operator chaining | On | On | Removes serialization between chained operators; never disable on the hot path |
| `pipeline.auto-watermark-interval` | 200 ms | 50–200 ms | Only matters for event-time operators |
| `pipeline.object-reuse` | false | true (if operators don't keep references) | Fewer allocations, less GC |
| `table.exec.mini-batch.enabled` | false | false | Mini-batch buffers rows → adds latency |
| Kafka source `fetch.max.wait.ms` + `fetch.min.bytes` | 500 ms + 1 byte | keep `fetch.min.bytes=1` | With min bytes = 1, fetches return as soon as data exists |
| Kafka sink `linger.ms` | client default (small) | 0–5 ms | Producer batching delay |
| Checkpoint interval | off | 10 s | Too frequent → competes with processing; too rare → long replays |

The buffer timeout deserves the most attention. With $h$ network hops (shuffles) on the path and a timeout $\tau$, the worst-case added delay at low load is:

$$
\Delta L_{\text{buffers}} \le h \cdot \tau
$$

P1's job has two shuffles (source → keyBy, OVER → sink): at the default $\tau = 100$ ms, up to **200 ms** of pure buffering; at $\tau = 5$ ms, up to 10 ms. At high load buffers fill before the timeout and the knob matters less — which is why the effect must be measured **across load steps**, not at a single rate.

The cost is throughput: smaller, more frequent network transfers mean more per-buffer overhead. If $c_b$ is the fixed cost per buffer flush and records per buffer drop from $n$ to $n'$, CPU per record grows roughly by $c_b\,(1/n' - 1/n)$. On a 4-core laptop that CPU is shared with everything else — another reason to measure.

### 2.3 Reading backpressure

Flink reports, per subtask, how each second is spent:

$$
\texttt{busyTimeMsPerSecond} + \texttt{backPressuredTimeMsPerSecond} + \texttt{idleTimeMsPerSecond} \approx 1000
$$

| Pattern | Interpretation |
|---|---|
| Operator X **busy ≈ 1000**, its upstream **backpressured** | X is the bottleneck (CPU-bound): optimize X or add parallelism |
| Sink busy low but **backpressured** high, external system slow | The external system is the bottleneck |
| Everything idle, Kafka lag growing | Source can't keep up: partitions/parallelism, or the job isn't consuming |
| One subtask busy, siblings idle | **Key skew** (note 01) |

Buffer pool metrics confirm it: `outPoolUsage` near 1.0 on an operator means its output buffers are full — the operator downstream is slow. The web UI's **BackPressure** tab and the per-vertex flame graphs (enable `rest.flamegraph.enabled`) show the same from the UI.

```mermaid
graph LR
    S[Source<br/>backpressured 900] --> W[OVER window<br/>busy 980] --> K[Kafka sink<br/>busy 40, idle 950]
    style W fill:#f96,stroke:#333
```

*Reading the diagram: the window operator is saturated (busy ≈ 980 ms/s); the source upstream is backpressured; the sink downstream is idle. Fix the window operator — not the sink.*

### 2.4 Metrics that belong on the dashboard

Exported with the Prometheus reporter, these feed the P1 dashboards ([[../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers|OpenTelemetry]] covers tracing; this is metrics):

| Metric | Why |
|---|---|
| `numRecordsInPerSecond` / `numRecordsOutPerSecond` per operator | Throughput and where records stop |
| `busyTimeMsPerSecond`, `backPressuredTimeMsPerSecond` | Bottleneck location |
| Kafka source `pendingRecords` / consumer lag | Is the job keeping up? |
| `lastCheckpointDuration`, `numberOfFailedCheckpoints` | Health of fault tolerance |
| `currentInputWatermark` (event-time jobs) | Stuck watermarks (idle sources) |
| JVM heap used, GC time | Memory pressure |

---

## 3. Production Reality

### Laptop-specific traps

- **CPU oversubscription:** slots don't isolate CPU. Parallelism 8 on a machine where Flink realistically gets 2 threads increases context switching and **raises** latency. Match parallelism to cores you can actually spare.
- **Thermal throttling:** a 45 W laptop CPU (like the i5-10300H) drops frequency when hot or on battery. Tail latencies jump without any configuration change. Benchmark plugged in, with a high-performance power plan, and record the conditions.
- **WSL2 memory limit:** Docker Desktop's VM is capped by `%UserProfile%\.wslconfig`. If the VM swaps, all latency numbers are invalid. Size containers with `mem_limit` so the sum stays under the VM limit.
- **Disk:** RocksDB and checkpoints write to Docker volumes. Keep them on a volume, not a bind mount into the Windows filesystem (bind mounts across the WSL boundary are slow).

### An experiment, not an opinion: E3

To show the effect of the buffer timeout, P1 runs a sweep:

| Run | `buffer-timeout` | Load steps (ev/s) | Repetitions | Output |
|---|---|---|---|---|
| E3-a | 100 ms (default) | 1k, 2k, 5k, 8k | 3 | p50/p95/p99 of the `flink` segment and end-to-end |
| E3-b | 10 ms | same | 3 | same |
| E3-c | 1 ms | same | 3 | same + CPU per event |

Expected shape: a large p95 drop from (a) to (b) at low load, a smaller one at high load (buffers fill anyway), and rising CPU at (c). The chart goes straight into the README — it teaches more than a paragraph.

Caso real: teams moving a feature pipeline from a throughput-oriented batch-like configuration to an online-serving SLA frequently find that "Flink is slow" was really the 100 ms buffer timeout multiplied across a few shuffles. Lowering it is usually the first and cheapest win; mini-batch and Python bundling are the next suspects.

### Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| TaskManager won't start | Inconsistent memory settings | Read the computed breakdown in the error; set only `process.size` first |
| `OutOfMemoryError: Direct buffer memory` | Network/off-heap too small for parallelism | Increase network memory fraction or reduce parallelism |
| Container `OOMKilled` | `mem_limit` < `process.size` + margin | Container limit ≥ process size |
| p95 floor ≈ n × 100 ms | Default buffer timeout | §2.2 |
| Latency spikes every few seconds | GC pauses or checkpoint upload competing for CPU | GC logs; incremental checkpoints; longer interval |
| Random tail spikes, no config change | Thermal throttling | §3 hygiene |

---

## 4. Code in Practice

### Laptop profile for the TaskManager (Flink 2.x `config.yaml` style)

```yaml
# conf/config.yaml (TaskManager) — keys verified against the pinned Flink version
taskmanager:
  numberOfTaskSlots: 4
  memory:
    process:
      size: 1200m            # the ONLY memory number set by hand; Flink derives the pools
execution:
  buffer-timeout: 5 ms       # default 100 ms: the single biggest latency knob
  checkpointing:
    interval: 10 s
    incremental: true
pipeline:
  object-reuse: true
  auto-watermark-interval: 200 ms
state:
  backend:
    type: rocksdb
metrics:
  reporter:
    prom:
      factory.class: org.apache.flink.metrics.prometheus.PrometheusReporterFactory
      port: 9249
rest:
  flamegraph:
    enabled: true
```

⚠️ **Warning:** depending on the image, the Prometheus reporter JAR may need to be copied from `opt/` into `plugins/` (or enabled via an environment variable). If `:9249/metrics` returns nothing, check `plugins/` first.

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: flink
    scrape_interval: 5s
    static_configs:
      - targets: ["jobmanager:9249", "taskmanager:9249"]
```

### Finding the bottleneck from the REST API

```python
# bottleneck.py — print busy/backpressure per vertex of the running job
import requests

BASE = "http://localhost:8081"
job = next(j for j in requests.get(f"{BASE}/jobs").json()["jobs"] if j["status"] == "RUNNING")
for v in requests.get(f"{BASE}/jobs/{job['id']}").json()["vertices"]:
    bp = requests.get(f"{BASE}/jobs/{job['id']}/vertices/{v['id']}/backpressure").json()
    level = bp.get("backpressureLevel", "n/a")   # ok / low / high
    print(f"{v['name'][:60]:<60} parallelism={v['parallelism']} backpressure={level}")
# The first operator whose UPSTREAM is 'high' while it is 'ok' is usually the culprit.
```

### ❌/✅ Parallelism and slots on a 4-core laptop

```yaml
# ❌ "More is faster": 12 slots, parallelism 12 on ~2 spare threads → context switching, worse p95
taskmanager.numberOfTaskSlots: 12
parallelism.default: 12

# ✅ Match real cores and Kafka partitions; scale out only when busy time says so
taskmanager.numberOfTaskSlots: 4
parallelism.default: 2
```

### 📦 Compression code: memory budget and the buffer-timeout trade-off

```python
# 📦 Compression code: TaskManager memory breakdown + buffer timeout latency/CPU model
# Covers: Flink memory pools, worst-case buffering delay, CPU cost of smaller buffers
def tm_breakdown(process_mb: int) -> dict:
    overhead = min(max(0.1 * process_mb, 192), 1024)
    metaspace = 256
    total_flink = process_mb - overhead - metaspace
    network = min(max(0.1 * total_flink, 64), 1024)
    managed = 0.4 * total_flink
    heap_and_offheap = total_flink - network - managed
    return {k: round(v) for k, v in dict(overhead=overhead, metaspace=metaspace, managed=managed,
            network=network, heap_and_offheap=heap_and_offheap).items()}

print("1200m TM:", tm_breakdown(1200))   # metaspace + overhead ≈ 37% of a small TM

hops, rate_per_channel = 2, 500  # 2 shuffles; records/s per network channel at low load
for tau_ms in (100, 10, 5, 1):
    worst = hops * tau_ms                               # buffers rarely fill at low load
    recs_per_flush = max(1, rate_per_channel * tau_ms / 1000)
    flushes_per_s = rate_per_channel / recs_per_flush
    print(f"buffer-timeout={tau_ms:>3} ms → worst added latency ≈ {worst:>3} ms, "
          f"~{flushes_per_s:.0f} flushes/s per channel")
# ¡Sorpresa! At low load the default 100 ms timeout alone can cost 200 ms on a 2-shuffle job.
```

---

## 🎯 Key Takeaways
- Set **`taskmanager.memory.process.size`** and let Flink derive the pools; metaspace + overhead dominate small TaskManagers.
- RocksDB and Python live in **managed memory**; shuffles live in **network memory** — size both deliberately.
- **`execution.buffer-timeout`** is the biggest latency knob: worst case ≈ hops × timeout at low load.
- Read **busy / backpressured / idle** per subtask: the bottleneck is busy while its upstream is backpressured.
- Export metrics to Prometheus; watch throughput, lag, checkpoint health, watermarks, GC.
- On laptops, CPU oversubscription, **thermal throttling**, and WSL2 swap invalidate benchmarks — control them.
- Prove each knob with a **sweep across load steps** (E3), not a single run.

## References
- Apache Flink — *Set up TaskManager Memory* and *Memory Tuning Guide*
- Apache Flink — *Monitoring Back Pressure*, *Metrics*, *Prometheus reporter*
- Apache Flink — *Configuration* reference (`execution.buffer-timeout`, checkpointing, state backend keys)
- Apache Kafka — consumer `fetch.min.bytes` / `fetch.max.wait.ms`, producer `linger.ms` documentation
- [[05 - PyFlink and the DataStream API|Previous: PyFlink and the DataStream API]] · [[07 - Capstone - Kafka to Flink to Redis Feature Pipeline|Next: Capstone]]
