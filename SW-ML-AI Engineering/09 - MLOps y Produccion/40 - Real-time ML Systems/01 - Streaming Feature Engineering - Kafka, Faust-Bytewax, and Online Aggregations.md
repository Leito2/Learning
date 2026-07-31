# 🎯 01 - Streaming Feature Engineering — Kafka, Faust/Bytewax, and Online Aggregations

> **The foundation of real-time ML. Compute features from event streams in <100ms. Online aggregations, windowed computations, and integration with Feast for online feature stores.**

## 🎯 Learning Objectives
- Set up Kafka locally with KRaft mode (no ZooKeeper)
- Use Faust/Bytewax for stream processing in Python
- Implement online aggregations (count, sum, mean, percentiles)
- Build windowed computations (tumbling, sliding, session)
- Integrate streaming features with Feast online feature store
- Handle late-arriving events and out-of-order processing
- Build fault-tolerant feature pipelines with checkpoints

## Introduction

The foundation of real-time ML is **streaming feature engineering**. Where batch ML computes features from a frozen dataset once a day, streaming feature engineering computes features from live events as they arrive.

Three operations dominate:

1. **Online aggregations** — count, sum, mean, percentiles over sliding windows
2. **Windowed computations** — count of events per user per 5 minutes, sum of transactions per account per hour
3. **Joins with state** — enrich events with user profile, account history, etc.

This note covers Kafka fundamentals, Faust/Bytewax for Python stream processing, online aggregations, windowed computations, and Feast integration.

![Streaming feature pipeline](https://example.com/streaming.png)

---

## 1. Kafka Fundamentals (2026 Edition)

Kafka is the dominant streaming platform. The 2024 KRaft release replaced ZooKeeper with the Raft consensus protocol — simpler deployment, better scalability.

### 1.1 The Kafka primitives

| Primitive | Purpose |
|----------|---------|
| **Broker** | Kafka server; one or more form a cluster |
| **Topic** | Stream of records; partitioned for parallelism |
| **Partition** | Ordered, immutable sequence of records within a topic |
| **Producer** | Writes records to topics |
| **Consumer** | Reads records from topics (one or more in a group) |
| **Consumer group** | Set of consumers that share work on a topic |

```bash
# Start a single-broker Kafka cluster (KRaft mode)
docker run -d --name kafka \
    -p 9092:9092 \
    -e KAFKA_CFG_NODE_ID=0 \
    -e KAFKA_CFG_PROCESS_ROLES=controller,broker \
    -e KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@localhost:9093 \
    -e KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
    -e KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    bitnami/kafka:3.7
```

### 1.2 Producer and consumer in Python

```python
from confluent_kafka import Producer, Consumer
import json


# Producer
producer = Producer({"bootstrap.servers": "localhost:9092"})


def send_event(topic: str, key: str, value: dict):
    producer.produce(
        topic,
        key=key.encode(),
        value=json.dumps(value).encode(),
    )
    producer.poll(0)  # trigger delivery callbacks


# Consumer
consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "feature-pipeline",
    "auto.offset.reset": "earliest",
})
consumer.subscribe(["user-events"])


def consume_events(timeout_ms: int = 1000):
    msg = consumer.poll(timeout_ms)
    if msg is None or msg.error():
        return None
    return json.loads(msg.value())
```

Use **confluent-kafka** (the official client) for production. `aiokafka` is fine for async code.

---

## 2. Bytewax — Python Stream Processing (Faust Replacement)

**Faust was archived in 2024**. The recommended replacement is **Bytewax**, which is actively maintained and supports modern Python (3.10+, async).

```bash
pip install bytewax
```

### 2.1 Basic Bytewax pipeline

```python
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.connectors.kafka import KafkaSource, KafkaSink
from bytewax import run_main


# Input: Kafka topic "user-events"
source = KafkaSource(
    brokers=["localhost:9092"],
    topics=["user-events"],
    starting_offset="earliest",
)

# Output: Kafka topic "user-features"
sink = KafkaSink(brokers=["localhost:9092"], topic="user-features")


flow = Dataflow("user-feature-pipeline")
events = op.input("input", flow, source)
parsed = op.map("parse", events, lambda x: json.loads(x.value))
features = op.map("extract_features", parsed, lambda e: {
    "user_id": e["user_id"],
    "event_type": e["event_type"],
    "timestamp": e["timestamp"],
})
op.output("output", features, sink)


if __name__ == "__main__":
    run_main(flow)
```

### 2.2 Stateful processing

The real value of stream processing is **state**: aggregations that remember past events.

```python
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.window import SystemClockConfig, TumblingWindowConfig


flow = Dataflow("click-counter")

# Define a 60-second tumbling window
clock_config = SystemClockConfig()
window_config = TumblingWindowConfig(length=60.0)  # 60 seconds

events = op.input("kafka_in", flow, KafkaSource(...))
keyed = op.key_on("key", events, lambda e: e["user_id"])
windowed = op.window_state(
    "60s_windows",
    keyed,
    clock_config,
    window_config,
)
counted = op.count("count_events", windowed)
op.output("kafka_out", counted, KafkaSink(...))
```

This counts events per user per 60 seconds. For each new event, the window's count is incremented.

### 2.3 Window types

| Window | Behavior |
|--------|----------|
| **Tumbling** | Fixed-size, non-overlapping (e.g., 0-60s, 60-120s) |
| **Sliding** | Fixed-size, overlapping (e.g., every 30s, 60s window) |
| **Session** | Event-driven; closes after N seconds of inactivity |
| **Global** | One window covering all time |

For real-time ML:
- **Tumbling**: hourly aggregations (count per user per hour)
- **Sliding**: rolling averages (5-minute moving average of clicks)
- **Session**: user session features (time spent, pages viewed)

---

## 3. Online Aggregations

### 3.1 The four aggregation types

```python
# 1. Count
event_count = count_events(window="5m")

# 2. Sum
spend_total = sum_field(field="amount", window="1h")

# 3. Mean (average)
avg_latency = mean_field(field="latency_ms", window="5m")

# 4. Percentile (p50, p95, p99)
p99_latency = percentile_field(field="latency_ms", percentile=99, window="5m")
```

### 3.2 Implementation in Bytewax

```python
from bytewax.operators import reduce


def aggregate_clicks(window_data):
    """Aggregate clicks per user in a 5-minute window."""
    user_id, events = window_data
    return {
        "user_id": user_id,
        "click_count": len(events),
        "unique_pages": len(set(e["page"] for e in events)),
        "total_value": sum(e.get("value", 0) for e in events),
        "first_event": min(e["timestamp"] for e in events),
        "last_event": max(e["timestamp"] for e in events),
    }


flow = Dataflow("click-aggregator")
events = op.input("kafka_in", flow, KafkaSource(...))
keyed = op.key_on("key", events, lambda e: e["user_id"])
windowed = op.window_state("5m", keyed, SystemClockConfig(), TumblingWindowConfig(length=300.0))
aggregated = op.map("aggregate", windowed, aggregate_clicks)
op.output("kafka_out", aggregated, KafkaSink(...))
```

### 3.3 Storing aggregated features to Feast

Feast (covered in [[09 - MLOps y Produccion/27 - Feast and Feature Stores|09/27 Feast]]) provides online feature storage. Push aggregated features to Feast from Bytewax:

```python
from feast import FeatureStore
import json


fs = FeatureStore(repo_path="feature_repo/")


def push_to_feast(aggregated):
    """Push aggregated features to Feast online store."""
    fs.write_to_online_store(
        feature_view_name="user_click_features",
        df=pd.DataFrame([{
            "user_id": aggregated["user_id"],
            "click_count_5m": aggregated["click_count"],
            "unique_pages_5m": aggregated["unique_pages"],
            "total_value_5m": aggregated["total_value"],
            "event_timestamp": pd.Timestamp.now(),
        }]),
    )


op.map("to_feast", aggregated, push_to_feast)
```

The online feature store is now updated. The inference service reads these features at request time via Feast's `get_online_features()` API.

---

## 4. Windowed Joins with State

Real-time ML often requires **joining events with state**. E.g., "user just clicked X; what did they view in the last 5 minutes?"

```python
from bytewax.operators import join


flow = Dataflow("user-context-join")
clicks = op.input("clicks", flow, KafkaSource(...))  # user_id, page, timestamp
views = op.input("views", flow, KafkaSource(...))    # user_id, page, timestamp

# Window both streams to 5 minutes
clock_config = SystemClockConfig()
window_config = TumblingWindowConfig(length=300.0)

clicks_keyed = op.key_on("clicks_key", clicks, lambda e: e["user_id"])
views_keyed = op.key_on("views_key", views, lambda e: e["user_id"])

clicks_windowed = op.window_state("clicks_w", clicks_keyed, clock_config, window_config)
views_windowed = op.window_state("views_w", views_keyed, clock_config, window_config)

# Join within the same window
joined = op.join("join", clicks_windowed, views_windowed)
op.output("output", joined, KafkaSink(...))
```

This pattern is fundamental for real-time ML: joining multiple event streams within time windows.

---

## 5. Late-Arriving Events and Watermarks

Real-world events arrive **out of order** (network delays, mobile app offline, etc.). Stream processors handle this with **watermarks**.

```python
from bytewax.window import EventClockConfig


# Use event-time processing with a 30-second late-arrival tolerance
clock_config = EventClockConfig(
    wait_for_system_duration=30.0,  # wait 30s before closing window
)
```

Watermarks:
- The stream processor tracks the "highest" timestamp seen so far
- Windows are closed when `current_time - watermark > window_size`
- Events arriving after window closure are dropped (or routed to a side channel)

For real-time ML:
- 5-minute windows with 30-second late-arrival tolerance is typical
- Adjust tolerance based on observed late-arrival distribution
- Very late events (>5 minutes) are typically dropped (logged for audit)

---

## 6. Fault Tolerance

Bytewax uses **checkpointing** for fault tolerance:

```python
from bytewax.checkpointing import CheckpointConfig


flow = Dataflow("fault-tolerant-pipeline")
# ... operators ...

# Checkpoint every 60 seconds to local disk
checkpoint_config = CheckpointConfig(
    interval=60_000,  # ms
    storage_dir="/var/bytewax/checkpoints",
)

run_main(
    flow,
    checkpoint_config=checkpoint_config,
)
```

If the worker crashes:
1. Restart from the last checkpoint
2. Replay events from the last committed offset
3. Resume processing

Kafka offsets are committed in the checkpoint. Combined with Kafka's `enable.auto.commit=False` and manual offset tracking, you get exactly-once semantics (with idempotent processing).

---

## 7. Real-World Example — Fraud Detection

A complete streaming feature pipeline for fraud detection:

```python
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.window import SystemClockConfig, TumblingWindowConfig


flow = Dataflow("fraud-features")

# Stream 1: transactions
transactions = op.input("transactions", flow, KafkaSource(
    brokers=["localhost:9092"], topics=["transactions"]
))
parsed_tx = op.map("parse_tx", transactions, lambda x: json.loads(x.value))

# Stream 2: account profile (CDC from PostgreSQL)
profiles = op.input("profiles", flow, KafkaSource(
    brokers=["localhost:9092"], topics=["account-profiles"]
))
parsed_profiles = op.map("parse_p", profiles, lambda x: json.loads(x.value))

# Compute velocity: transactions per account in last 5 minutes
tx_keyed = op.key_on("by_account", parsed_tx, lambda e: e["account_id"])
tx_windowed = op.window_state(
    "5min_windows", tx_keyed,
    SystemClockConfig(),
    TumblingWindowConfig(length=300.0),
)
tx_velocity = op.map("velocity", tx_windowed, lambda w: {
    "account_id": w[0],
    "tx_count_5m": len(w[1]),
    "total_amount_5m": sum(e["amount"] for e in w[1]),
    "avg_amount_5m": sum(e["amount"] for e in w[1]) / len(w[1]),
})

# Enrich with profile
features = op.map("enrich", tx_velocity, lambda w: {
    **w[1],
    "timestamp": pd.Timestamp.now(),
})

# Push to Feast and to inference topic
op.output("feast", features, push_to_feast_op)
op.output("infer", features, KafkaSink(brokers=["localhost:9092"], topic="fraud-features"))
```

The result: for every transaction, the model has access to:
- The transaction itself
- The 5-minute velocity (count, total, average)
- The account profile (from CDC)

All in <100ms.

---

## 8. Antipatterns

### 8.1 Antipattern 1: Synchronous aggregation in API handlers

```python
# ❌ Compute aggregates synchronously in API request — slow and blocking
@app.post("/predict")
async def predict(request):
    recent = await db.query("SELECT * FROM events WHERE user_id = $1")
    avg = sum(e["amount"] for e in recent) / len(recent)  # slow
    return model.predict(features={"avg_amount": avg})

# ✅ Aggregates computed in stream pipeline; API just reads from Feast
@app.post("/predict")
async def predict(request):
    features = fs.get_online_features(
        feature_view="user_features",
        entity_rows=[{"user_id": request.user_id}],
    ).to_dict()
    return model.predict(features=features)
```

### 8.2 Antipattern 2: Storing aggregations in the database

```python
# ❌ Write aggregates back to Postgres, read them on inference
# Adds DB write latency, DB read latency, and consistency issues
db.execute("UPDATE users SET click_count_5m = $1 WHERE id = $2", count, user_id)

# ✅ Aggregates go directly to Feast online store (Redis or DynamoDB)
fs.write_to_online_store("user_features", aggregated_df)
```

### 8.3 Antipattern 3: Unbounded windows

```python
# ❌ One window from epoch to now — never closes, unbounded memory
window_config = GlobalWindowConfig()

# ✅ Tumbling or sliding windows with explicit length
window_config = TumblingWindowConfig(length=300.0)  # 5 minutes
```

### 8.4 Antipattern 4: No late-arrival handling

```python
# ❌ Drop events that arrive after window closes
windowed = op.window_state(...)
# No late-arrival config → late events silently dropped

# ✅ Configure watermark tolerance
clock_config = EventClockConfig(wait_for_system_duration=30.0)
windowed = op.window_state(..., clock_config, ...)
```

### 8.5 Antipattern 5: Synchronous Feast writes

```python
# ❌ Synchronous Feast writes block stream processing
op.map("to_feast", features, lambda f: fs.write_to_online_store(...))

# ✅ Async batch writes
op.map("batch_feast", features, batch_writes_to_feast)  # accumulates 1000 events, writes batch
```

---

## 🎯 Key Takeaways

- Kafka 2026 uses KRaft mode (no ZooKeeper); simpler, more scalable.
- **Bytewax** replaced Faust as the Python stream processing standard.
- Four aggregation types: count, sum, mean, percentile — all computable online.
- Three window types: tumbling (fixed), sliding (overlapping), session (event-driven).
- Online feature stores (Feast, Tecton) integrate with stream pipelines for sub-100ms reads.
- Late-arrival handling via watermarks; tune tolerance based on observed lag.
- Fault tolerance via checkpointing; combine with manual Kafka offset commit for exactly-once.
- Avoid sync aggregation in API, storing aggregates in DB, unbounded windows, no late handling, sync Feast writes.

## References

- Kafka KRaft — [kafka.apache.org/documentation/#kraft](https://kafka.apache.org/documentation/#kraft)
- Bytewax — [bytewax.io](https://bytewax.io/)
- Faust archived — [github.com/faust-streaming/faust](https://github.com/faust-streaming/faust)
- Feast — [feast.dev](https://feast.dev/)
- [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/01 - Apache Kafka|Kafka note 01]] — Kafka basics
- [[09 - MLOps y Produccion/27 - Feast and Feature Stores|Feast course]] — online feature store
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/02 - Online Inference and Event-Driven ML|Note 02 — Online Inference]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/03 - Change Data Capture for ML|Note 03 — CDC]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/05 - Capstone - Production Real-time ML Pipeline|Note 05 — Capstone]]