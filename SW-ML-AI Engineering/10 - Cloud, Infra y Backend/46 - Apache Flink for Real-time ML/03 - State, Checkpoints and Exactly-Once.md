# 💾 03 - State, Checkpoints and Exactly-Once

A streaming feature pipeline is a machine that **remembers**: every user's last 10 minutes of payments, every device seen in the last day, every running sum. When a TaskManager dies at 3 a.m., the question is not whether you lose that memory, but whether you get it back **without losing events and without counting any event twice** — because a double-counted payment is a false fraud signal.

## 🎯 Learning Objectives
- Explain **keyed state** and its types (Value, List, Map, Aggregating) and **operator state**
- Compare state backends: **HashMap (heap)**, **RocksDB**, and Flink 2.x **disaggregated state**
- Understand **asynchronous barrier snapshotting**: barriers, alignment, and unaligned checkpoints
- Distinguish **at-most-once, at-least-once, exactly-once state**, and **end-to-end exactly-once**
- Quantify the **latency cost** of transactional (two-phase commit) sinks
- Use **idempotent sinks** to get effectively-once results without transactions
- Use savepoints, TTL, and operator UIDs to evolve and operate stateful jobs safely

## Introduction

Stateless streaming is easy: map, filter, forward. Everything interesting in ML features — counts, sums, distinct sets, sequences — is **stateful**. Flink keeps that state **local to the subtask** that owns each key (thanks to `keyBy`, note 01), which is what makes per-event updates fast: no network round-trip to an external database for every event.

Local state creates the fault-tolerance problem. A crashed process takes its memory with it. Flink's answer, inherited from distributed systems research, is to periodically take a **globally consistent snapshot** of all operator state *together with the source offsets* that produced it, without stopping the stream. On failure, every operator restores the last snapshot and the sources rewind to the matching offsets — the job resumes as if the failure never happened, at least as far as **state** is concerned.

What happens to the **outputs** already written before the crash is a separate question, and the answer determines your latency. That distinction — exactly-once *state* vs exactly-once *end-to-end* — is the single most misunderstood topic in stream processing, and it drives one of P1's central design decisions.

---

## 1. The Problem and Why This Solution Exists

### Why not keep state in an external database?

The naive design reads and writes state in Redis or Postgres for every event. It has three problems:

1. **Latency:** one or more network round-trips per event, per feature.
2. **Consistency on failure:** if the job crashes after updating the database but before committing the Kafka offset, the event is replayed and **counted twice**. Coordinating database writes with source offsets requires distributed transactions.
3. **Throughput:** the database becomes the bottleneck for every streaming job that touches it.

### Distributed snapshots: from Chandy–Lamport to Flink

In 1985, Chandy and Lamport described how to record a **consistent global state** of a distributed system without stopping it, by sending marker messages through channels. Flink adapted this as **Asynchronous Barrier Snapshotting** (Carbone et al., *Lightweight Asynchronous Snapshots for Distributed Dataflows*, 2015): instead of markers on every channel, the sources inject **checkpoint barriers** into the stream, and each operator snapshots its state when the barrier passes.

The result: a snapshot that contains, for checkpoint $n$, the state of every operator **after processing exactly the records before barrier $n$**, plus the source offsets at barrier $n$. That pairing is what makes recovery consistent.

---

## 2. Conceptual Deep Dive

### 2.1 Keyed state and operator state

![State partitioning](https://nightlies.apache.org/flink/flink-docs-stable/fig/state_partitioning.svg)

*Figure: keyed state is partitioned along with the stream — each subtask holds state only for its keys. Source: Apache Flink documentation.*

| State type            | Scope                    | Example for ML features                                  |
|-----------------------|--------------------------|----------------------------------------------------------|
| `ValueState<T>`       | One value per key        | Last payment timestamp per user (`secs_since_last`)      |
| `ListState<T>`        | List per key             | Buffer of recent events (for custom windows)             |
| `MapState<K, V>`      | Map per key              | Countries seen in the last hour → last seen time         |
| `AggregatingState`    | Incremental aggregate    | Running sum/count with add/merge logic                   |
| Operator state        | Per subtask, not per key | Kafka source partition offsets                           |
| Broadcast state       | Replicated to all subtasks | Fraud rules or thresholds updated at runtime           |

In Flink SQL you never declare state explicitly — the planner creates it. An `OVER` window with `RANGE INTERVAL '10' MINUTE` keeps, per user, the rows inside the range (or incremental accumulators plus retraction data), and cleans them up as time advances.

### 2.2 State backends

| Backend                        | Where state lives                    | Max state size        | Access cost             | Snapshot                     |
|--------------------------------|--------------------------------------|-----------------------|-------------------------|------------------------------|
| **HashMap** (heap)             | JVM heap objects                     | Bounded by heap       | Fastest (object access) | Full copy, can be large      |
| **EmbeddedRocksDB**            | Local disk + RocksDB block cache in managed memory | Bounded by disk | Serialize/deserialize per access | **Incremental** (only changed SST files) |
| **Disaggregated (ForSt, 2.x)** | Remote/object storage with local cache | Very large          | Higher, cache-dependent | Cheap, near-instant restore  |

For a laptop-sized fraud pipeline, **RocksDB with incremental checkpoints** is the default: state size is bounded by disk instead of the 1.2 GB TaskManager budget, and checkpoints only upload what changed. The cost is serialization on every state access — measurable, but typically microseconds.

⚠️ **Warning:** RocksDB's memory (block cache, write buffers) comes from Flink **managed memory**. If you shrink `taskmanager.memory.process.size` without understanding the memory model, RocksDB starves and latency spikes. Note 06 walks through the TaskManager memory breakdown.

💡 **Tip:** Disaggregated state (introduced in Flink 2.0) targets cloud-native deployments where local disks are small and fast rescaling matters. Treat it as the direction of travel for large production jobs, not as the default for a laptop project; verify its maturity in the Flink version you pin.

### 2.3 Checkpoint barriers and alignment

![Checkpoint barriers](https://nightlies.apache.org/flink/flink-docs-stable/fig/stream_barriers.svg)

*Figure: barriers flow with the records and split the stream into "before checkpoint n" and "after". Source: Apache Flink documentation.*

1. The JobMaster triggers checkpoint $n$; each source records its offsets and emits barrier $n$.
2. An operator with one input snapshots its state when barrier $n$ arrives and forwards the barrier.
3. An operator with **several inputs** must **align**: once barrier $n$ arrives on one input, it buffers that input's further records until barrier $n$ arrives on all inputs, then snapshots.

![Barrier alignment](https://nightlies.apache.org/flink/flink-docs-stable/fig/stream_aligning.svg)

*Figure: aligned checkpoints pause the inputs that are ahead. Source: Apache Flink documentation.*

Alignment under backpressure is slow: barriers queue behind buffered records, checkpoints take longer, and in extreme cases time out. **Unaligned checkpoints** let barriers overtake buffered records and include those in-flight buffers in the snapshot — faster checkpoints under backpressure at the cost of larger snapshots.

The snapshot itself is **asynchronous**: the operator takes a cheap synchronous copy (or a RocksDB snapshot) and uploads it in the background while processing continues.

![Checkpointing](https://nightlies.apache.org/flink/flink-docs-stable/fig/checkpointing.svg)

*Figure: each operator snapshots its state to durable storage; the checkpoint completes when all acknowledge. Source: Apache Flink documentation.*

### 2.4 Delivery guarantees, precisely

| Guarantee                  | Meaning                                                             | How                                                      |
|----------------------------|---------------------------------------------------------------------|----------------------------------------------------------|
| At-most-once               | Each event affects results 0 or 1 times                              | No checkpoints, no replay                                |
| At-least-once              | ≥ 1 times; duplicates possible after recovery                        | Checkpoints without alignment, or non-transactional sinks |
| **Exactly-once state**     | Each event affects **Flink's internal state** exactly once           | Aligned (or unaligned) checkpoints + source rewind       |
| **End-to-end exactly-once** | Each event affects **external outputs** exactly once                | Exactly-once state **+ transactional or idempotent sink** |

After a failure, Flink restores state from checkpoint $n$ and replays records since barrier $n$. Internal state is correct. But the sink may have **already written** outputs for some of those replayed records before the crash. Without help from the sink, downstream systems see duplicates.

### 2.5 Two-phase commit sinks and their latency cost

The Kafka sink offers `exactly-once` delivery using Kafka **transactions**: records written between two checkpoints belong to one transaction, which is **pre-committed** at the barrier and **committed** only when the checkpoint completes. Consumers using `isolation.level=read_committed` only see committed data.

That means an output record becomes visible only after the next checkpoint completes. With checkpoint interval $T_{cp}$ and checkpoint duration $D_{cp}$, a record produced at a uniformly random moment waits:

$$
L_{\text{visible}} \approx L_{\text{process}} + U(0,\, T_{cp}) + D_{cp}
\qquad
\mathbb{E}[L_{\text{visible}}] \approx L_{\text{process}} + \frac{T_{cp}}{2} + D_{cp}
$$

With $T_{cp} = 1\,\text{s}$ and $D_{cp} = 200\,\text{ms}$, the **average** added latency is ~700 ms and the p95 is above 1.1 s. For a p95 ≤ 80 ms fraud pipeline this is disqualifying — which is why P1 uses **at-least-once + idempotency** on the hot path and measures exactly-once only as an experiment.

### 2.6 Idempotent sinks: effectively-once without transactions

If writing the same output twice produces the same final result, duplicates are harmless:

- **Upserts by key:** Redis `HSET features:{user_id} …`, Postgres `INSERT … ON CONFLICT (payment_id) DO NOTHING`, Qdrant points with deterministic IDs.
- **Deduplication downstream:** consumers keep a short-lived set of processed `payment_id`s.

Formally, a sink operation $f$ is idempotent if $f(f(s, x), x) = f(s, x)$. With at-least-once delivery and idempotent sinks, the **observable** result equals exactly-once — with no transaction latency. This is the pattern most low-latency production systems actually use.

### 2.7 Recovery time

After a crash the job restarts, restores state, rewinds sources to the checkpoint offsets, and must **catch up** on the backlog accumulated since the checkpoint plus during downtime. With input rate $\lambda$, processing capacity $\mu > \lambda$, time since the last checkpoint $t_{cp}$, and restart time $T_r$:

$$
T_{\text{recover}} \approx T_r + \frac{\lambda\,(t_{cp} + T_r)}{\mu - \lambda}
$$

At $\lambda = 5{,}000$ ev/s, $\mu = 8{,}000$ ev/s, $t_{cp} = 10$ s and $T_r = 15$ s, catching up takes $\approx 15 + \frac{5000 \cdot 25}{3000} \approx 57$ s. Two levers: shorter checkpoint intervals (less to replay) and **headroom** ($\mu \gg \lambda$). A pipeline running at 95% of capacity takes a very long time to recover — capacity planning is a resilience decision.

---

## 3. Production Reality

### Estimating state size

For an `OVER` window over 10 minutes keyed by user, the state roughly holds the rows inside the range:

$$
S \approx N_{\text{active users}} \times \bar{r}_{10\text{min}} \times b_{\text{row}}
$$

100k active users × 3 payments per 10 min × ~200 bytes ≈ 60 MB — trivial for RocksDB. A 24-hour distinct-devices feature over the same users can be 100× larger. Always estimate before choosing window ranges, and use **state TTL** so inactive keys expire.

### Operating stateful jobs

- **Assign stable operator UIDs** (DataStream `.uid("features-10m")`). Savepoints map state to operators by UID; auto-generated IDs change when you edit the job, and the restore fails.
- **Savepoints vs checkpoints:** checkpoints are automatic and owned by Flink (for failure recovery); savepoints are manual, portable snapshots you take **before upgrades, rescaling, or Flink version changes**.
- **Checkpoint interval:** 10 s is a sane laptop default — frequent enough for short replays, rare enough not to compete with processing. Monitor checkpoint duration; if it approaches the interval, something is wrong (backpressure, state too large, slow storage).

Caso real: in payment systems, a classic incident is a feature job restarted with at-least-once delivery feeding a **non-idempotent** counter in an external store — every replay inflates the counts, users suddenly look like they made dozens of payments, and the fraud model blocks legitimate customers after each deploy. The fix is never "avoid restarts"; it is making the sink idempotent (keyed upserts) or the delivery transactional.

### Known failure modes

| Symptom                                   | Cause                                         | Fix                                                |
|-------------------------------------------|-----------------------------------------------|----------------------------------------------------|
| Checkpoints time out                      | Backpressure delays barriers                  | Unaligned checkpoints; fix the slow operator       |
| State grows forever                       | No TTL on keys that stop appearing            | State TTL / window ranges                          |
| Savepoint restore fails after a change    | Operator IDs changed                          | Explicit UIDs from day one                         |
| Duplicates downstream after recovery      | At-least-once + non-idempotent sink           | Upserts by key or transactional sink               |
| Read-committed consumers see 1 s+ latency | Exactly-once Kafka sink                       | Expected — that's the 2PC cost (§2.5)              |

---

## 4. Code in Practice

### Checkpointing and state backend (SQL client / job config)

```sql
SET 'execution.checkpointing.interval' = '10 s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';     -- exactly-once STATE (aligned barriers)
SET 'execution.checkpointing.unaligned.enabled' = 'true'; -- survive backpressure
SET 'state.backend.type' = 'rocksdb';
SET 'execution.checkpointing.incremental' = 'true';
SET 'execution.checkpointing.dir' = 'file:///flink-checkpoints';  -- a volume; S3/GCS in production
SET 'table.exec.state.ttl' = '25 h';                     -- longest feature range + margin
```

⚠️ **Warning:** configuration keys were renamed across Flink versions (e.g., state backend and checkpoint storage options in 2.x). Copy keys from the documentation of the **exact** version you pin.

### Sink guarantees: the hot path vs the experiment

```sql
-- ✅ Hot path (P1): at-least-once to Kafka; consumers deduplicate by payment_id
CREATE TABLE payments_enriched (
  payment_id STRING, user_id STRING, amount DOUBLE,
  f_cnt_10m BIGINT, f_sum_10m DOUBLE
) WITH (
  'connector' = 'kafka',
  'topic' = 'payments-enriched',
  'properties.bootstrap.servers' = 'kafka:29092',
  'format' = 'json',
  'sink.delivery-guarantee' = 'at-least-once'
);

-- 🧪 Experiment E5: exactly-once end-to-end — consumers must use isolation.level=read_committed
--    and will only see records after each checkpoint completes (≈ T_cp/2 + D_cp extra latency).
--   'sink.delivery-guarantee' = 'exactly-once',
--   'sink.transactional-id-prefix' = 'fraud-features'
```

### Idempotent consumption downstream

```python
# ❌ Non-idempotent: a replay after recovery double-counts
redis.hincrby(f"stats:{user_id}", "payments", 1)

# ✅ Idempotent: the last write wins with the same value; replays are harmless
redis.hset(f"vel:{user_id}", mapping={"f_cnt_10m": f_cnt_10m, "f_sum_10m": f_sum_10m})

# ✅ Idempotent insert for the audit log
cur.execute("INSERT INTO decisions (payment_id, decision) VALUES (%s, %s) "
            "ON CONFLICT (payment_id) DO NOTHING", (payment_id, decision))
```

### 📦 Compression code: replays, idempotency, and the 2PC latency tax

```python
# 📦 Compression code: at-least-once replay vs idempotent sink, transactional visibility, recovery time
import random
import statistics

events = [f"p{i}" for i in range(1, 11)]
checkpoint_after = 6          # checkpoint n covered p1..p6
crash_after = 9               # sink already wrote p1..p9, then the TaskManager died

counter = 0                   # ❌ non-idempotent sink: increments
seen = {}                     # ✅ idempotent sink: keyed upsert
for e in events[:crash_after]:
    counter += 1
    seen[e] = 1
for e in events[checkpoint_after:]:   # recovery: rewind to checkpoint, replay p7..p10
    counter += 1
    seen[e] = 1
print(f"non-idempotent count = {counter} (true = 10)  |  idempotent count = {len(seen)}")
# ¡Sorpresa! Exactly-once STATE inside Flink did not prevent duplicates OUTSIDE it.

T_cp, D_cp, L_proc = 1.0, 0.2, 0.015  # seconds
visible = [L_proc + random.uniform(0, T_cp) + D_cp for _ in range(100_000)]
p95 = statistics.quantiles(visible, n=100)[94]
print(f"exactly-once Kafka sink: mean={statistics.mean(visible)*1000:.0f} ms, p95={p95*1000:.0f} ms")

lam, mu, t_cp, T_r = 5_000, 8_000, 10, 15
print(f"recovery ≈ {T_r + lam * (t_cp + T_r) / (mu - lam):.0f} s at {lam/mu:.0%} utilization")
```

---

## 🎯 Key Takeaways
- Flink keeps **keyed state local** to the subtask owning each key — fast, but it must be snapshotted to survive failures.
- **RocksDB + incremental checkpoints** bounds state by disk, not heap; it costs serialization per access and needs managed memory.
- **Barriers** split the stream; aligned snapshots give exactly-once *state*; **unaligned** checkpoints keep working under backpressure.
- Exactly-once **state** ≠ exactly-once **outputs**: replays after recovery can duplicate external writes.
- Transactional (2PC) Kafka sinks add ≈ $T_{cp}/2 + D_{cp}$ of visibility latency — incompatible with tight p95 budgets.
- **At-least-once + idempotent sinks** (keyed upserts, `ON CONFLICT`) give effectively-once results with no transaction latency.
- Plan **headroom** ($\mu \gg \lambda$) and stable **operator UIDs**: they decide how fast and how safely you recover.

## References
- Carbone et al., *Lightweight Asynchronous Snapshots for Distributed Dataflows* (arXiv:1506.08603, 2015)
- Chandy & Lamport, *Distributed Snapshots: Determining Global States of Distributed Systems* (ACM TOCS, 1985)
- Apache Flink — *Stateful Stream Processing*, *Checkpointing*, *State Backends*, *Kafka connector* docs
- Apache Flink 2.0 release announcement — disaggregated state (ForSt)
- [[02 - Time, Watermarks and Windows|Previous: Time, Watermarks and Windows]] · [[04 - Flink SQL for Real-time Features|Next: Flink SQL for Real-time Features]]
