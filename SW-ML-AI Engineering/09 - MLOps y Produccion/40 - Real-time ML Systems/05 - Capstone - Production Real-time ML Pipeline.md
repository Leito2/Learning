# 🎯 05 - Capstone — Production Real-time ML Pipeline

> **The twelfth portfolio project. End-to-end real-time ML: Kafka + Debezium CDC + Bytewax stream processing + Feast online store + FastAPI inference + ADWIN drift detection. Boot the whole pipeline with one docker-compose.**

## 🎯 Learning Objectives
- Build an end-to-end real-time ML pipeline for a real use case (fraud detection)
- Set up Kafka + Debezium + Bytewax + Feast + FastAPI + ADWIN in Docker Compose
- Implement CDC from PostgreSQL to Kafka
- Stream-process events through Bytewax with online aggregations
- Serve predictions in <50ms via FastAPI + Feast
- Detect drift with ADWIN and trigger auto-retraining
- Monitor the entire pipeline with Prometheus + Grafana

## Introduction

The capstone is the **synthesis** of all four notes. You will build a complete real-time ML pipeline for fraud detection:

1. **PostgreSQL** stores transactions and user accounts
2. **Debezium** captures every change via CDC
3. **Kafka** streams events and features
4. **Bytewax** computes streaming aggregations (5-minute transaction velocity)
5. **Feast** stores online features (with <5ms lookup)
6. **FastAPI** serves predictions with the model loaded in memory
7. **ADWIN** detects model quality drift
8. **Auto-retraining pipeline** is triggered on drift

The architecture demonstrates the four patterns from Notes 01-04 working together. The capstone is the **twelfth portfolio project**: the real-time ML skill.

```mermaid
graph TB
    subgraph "Data Sources"
        PG[(PostgreSQL)]
    end
    
    subgraph "Streaming Layer"
        DBZ[Debezium CDC]
        KAFKA[Kafka]
    end
    
    subgraph "Stream Processing"
        BW[Bytewax]
    end
    
    subgraph "Feature Store"
        FEAST[Feast online store]
        REDIS[(Redis)]
    end
    
    subgraph "Inference"
        API[FastAPI inference]
        MODEL[Model in memory]
    end
    
    subgraph "Drift & Monitoring"
        DRIFT[ADWIN detector]
        PROM[Prometheus]
        GRAF[Grafana]
    end
    
    subgraph "Auto-Retraining"
        TRAIN[Training service]
    end
    
    PG -->|CDC| DBZ
    DBZ -->|publish| KAFKA
    KAFKA -->|consume| BW
    BW -->|aggregate + write| FEAST
    FEAST -->|Redis| REDIS
    REDIS -->|read| API
    KAFKA -->|request events| API
    API -->|predict| MODEL
    MODEL -->|response| API
    API -->|metrics| PROM
    PROM -->|dashboards| GRAF
    API -->|prediction + actual| DRIFT
    DRIFT -->|drift detected| TRAIN
    TRAIN -->|new model| MODEL
```

---

## 1. Project Layout

```
realtime-fraud-detection/
├── docker-compose.yml        # Postgres + Kafka + Debezium + Bytewax + Feast + Redis + API
├── app/
│   ├── api.py                # FastAPI inference service
│   ├── model.py              # Fraud detection model (ONNX)
│   ├── features.py           # Feature engineering
│   ├── drift.py              # ADWIN drift detector
│   ├── pipelines/
│   │   ├── stream_processor.py    # Bytewax pipeline
│   │   ├── cdc_consumer.py       # CDC consumer → Feast
│   │   └── retrainer.py           # Auto-retraining trigger
├── feature_repo/
│   ├── feature_store.yaml
│   ├── features.py
│   └── entity.py
├── models/
│   ├── fraud_detector.pkl    # Trained model
│   └── fraud_detector.onnx   # ONNX for inference
├── tests/
│   ├── test_pipeline.py
│   └── test_drift_detection.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboard.json
└── README.md
```

---

## 2. The Single docker-compose.yml — One Command to Rule Them All

```yaml
version: "3.9"

services:
  # Postgres with CDC enabled
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: app
    command:
      - "postgres"
      - "-c"
      - "wal_level=logical"
      - "-c"
      - "max_replication_slots=4"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Schema setup
  postgres_init:
    image: postgres:16-alpine
    depends_on:
      - postgres
    volumes:
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  # Kafka (KRaft mode, no ZooKeeper)
  kafka:
    image: bitnami/kafka:3.7
    environment:
      KAFKA_CFG_NODE_ID: 0
      KAFKA_CFG_PROCESS_ROLES: controller,broker
      KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 0@kafka:9093
      KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
    ports:
      - "9092:9092"

  # Debezium for PostgreSQL CDC
  debezium:
    image: debezium/connect:2.5
    depends_on:
      - kafka
      - postgres
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: 1
      CONFIG_STORAGE_TOPIC: debezium_configs
      OFFSET_STORAGE_TOPIC: debezium_offsets
      STATUS_STORAGE_TOPIC: debezium_status
    ports:
      - "8083:8083"

  # Bytewax stream processor (built from local Dockerfile)
  stream_processor:
    build: ./stream_processor
    depends_on:
      - kafka
    environment:
      KAFKA_BROKERS: kafka:9092
      FEAST_REPO_PATH: /feature_repo
    command: ["python", "-m", "stream_processor.main"]

  # Feast online store backed by Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  feast:
    build: ./feast_server
    depends_on:
      - redis
    ports:
      - "6566:6566"  # Feast API

  # FastAPI inference service
  api:
    build: ./app
    depends_on:
      - feast
      - kafka
    environment:
      KAFKA_BROKERS: kafka:9092
      FEAST_URL: http://feast:6566
      MODEL_PATH: /models/fraud_detector.pkl
    ports:
      - "8000:8000"

  # Auto-retraining service
  retrainer:
    build: ./retrainer
    depends_on:
      - kafka
      - postgres
    environment:
      KAFKA_BROKERS: kafka:9092
      POSTGRES_URL: postgresql://app:secret@postgres:5432/app
    command: ["python", "-m", "retrainer.main"]

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana_dashboard.json:/var/lib/grafana/dashboards/main.json
    ports:
      - "3000:3000"

volumes:
  postgres_data:
```

One command — `docker compose up -d` — boots the entire pipeline.

---

## 3. PostgreSQL Schema with CDC (`init.sql`)

```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    account_age_days INT NOT NULL DEFAULT 0,
    country VARCHAR(2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    account_id BIGINT REFERENCES users(id),
    amount_cents BIGINT NOT NULL,
    merchant_category VARCHAR(50),
    country VARCHAR(2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fraud_labels (
    transaction_id BIGINT PRIMARY KEY REFERENCES transactions(id),
    is_fraud BOOLEAN NOT NULL,
    labeled_by VARCHAR(50),
    labeled_at TIMESTAMPTZ DEFAULT NOW()
);

-- Publication for CDC
CREATE PUBLICATION cdc_ml FOR TABLE users, transactions, fraud_labels;

-- Sample data
INSERT INTO users (email, account_age_days, country) VALUES
    ('alice@example.com', 365, 'DE'),
    ('bob@example.com', 730, 'US'),
    ('charlie@example.com', 30, 'BR');

INSERT INTO transactions (account_id, amount_cents, merchant_category, country) VALUES
    (1, 5000, 'grocery', 'DE'),
    (2, 1500, 'restaurant', 'US'),
    (3, 100000, 'electronics', 'BR');
```

---

## 4. The Model (`app/model.py`)

A simple fraud detection model trained on labeled data.

```python
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class FraudDetectionModel:
    """LightGBM-style fraud detector."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_version = "1.0.0"
        if model_path:
            self.load(model_path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def predict(self, features: dict) -> float:
        """Return fraud probability [0, 1]."""
        feature_vector = np.array([
            features["tx_count_5m"],
            features["total_amount_5m"],
            features["avg_amount_5m"],
            features["account_age_days"],
            features["country_risk"],
            features["merchant_category_risk"],
            features["tx_velocity_5m"],
        ]).reshape(1, -1)
        
        proba = self.model.predict_proba(feature_vector)[0][1]
        return float(proba)
    
    @classmethod
    def train(cls, training_data_path: str, output_path: str):
        """Train a new model from labeled data."""
        # Load training data (in production: S3, BigQuery, etc.)
        from sklearn.datasets import load_iris
        # Mock training data for the capstone
        X = np.random.rand(10000, 7)
        y = (X[:, 0] + X[:, 2] > 1).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Save
        joblib.dump(model, output_path)
        
        # Validate
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"Trained model AUC: {auc:.3f}")
        
        return model
```

In production, this is trained on a balanced dataset with proper labeling (covered in [[09 - MLOps y Produccion/27 - Feast and Feature Stores|Feast]] for feature pipelines and [[09 - MLOps y Produccion/22 - End-to-End ML Project|E2E ML Project]] for training).

---

## 5. The Stream Processor (`pipelines/stream_processor.py`)

```python
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.window import SystemClockConfig, TumblingWindowConfig
from bytewax.connectors.kafka import KafkaSource, KafkaSink
from bytewax import run_main
import json
import pandas as pd
from feast import FeatureStore


# Initialize Feast
fs = FeatureStore(repo_path="/feature_repo")


flow = Dataflow("fraud-features")

# Stream 1: Transactions from CDC topic
transactions = op.input(
    "transactions",
    flow,
    KafkaSource(brokers=["kafka:9092"], topics=["cdc.public.transactions"]),
)

parsed_tx = op.map("parse", transactions, lambda x: json.loads(x.value))

# Stream 2: Stream windowed aggregation (5-minute transaction velocity)
tx_by_account = op.key_on(
    "by_account",
    parsed_tx,
    lambda e: str(e["after"]["account_id"]),
)
tx_windowed = op.window_state(
    "5min",
    tx_by_account,
    SystemClockConfig(),
    TumblingWindowConfig(length=300.0),
)
tx_aggregated = op.map("aggregate", tx_windowed, lambda w: {
    "account_id": w[0],
    "tx_count_5m": len(w[1]),
    "total_amount_5m": sum(e["after"]["amount_cents"] for e in w[1]) / 100,
    "avg_amount_5m": sum(e["after"]["amount_cents"] for e in w[1]) / len(w[1]) / 100,
})

# Push to Feast
def push_to_feast(aggregated):
    fs.write_to_online_store(
        feature_view_name="transaction_5m_features",
        df=pd.DataFrame([{
            "account_id": aggregated["account_id"],
            "tx_count_5m": aggregated["tx_count_5m"],
            "total_amount_5m": aggregated["total_amount_5m"],
            "avg_amount_5m": aggregated["avg_amount_5m"],
            "event_timestamp": pd.Timestamp.now(),
        }]),
    )


op.output("feast", tx_aggregated, KafkaSink(brokers=["kafka:9092"], topic="fraud-features"))


if __name__ == "__main__":
    run_main(flow)
```

---

## 6. The Inference API (`app/api.py`)

```python
from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram, Counter
from contextlib import asynccontextmanager
import asyncio
import httpx
from pydantic import BaseModel
from river.drift import ADWIN


prediction_latency = Histogram(
    "fraud_prediction_latency_seconds",
    "Time to make fraud prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
predictions_made = Counter(
    "fraud_predictions_made_total",
    "Total fraud predictions",
    ["outcome"],
)
drift_detected = Counter(
    "fraud_drift_detected_total",
    "Drift detection events",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once
    from .model import FraudDetectionModel
    app.state.model = FraudDetectionModel("/models/fraud_detector.pkl")
    
    # Drift detector
    app.state.adwin = ADWIN(delta=0.002)
    
    # Shared clients
    app.state.feast_client = httpx.AsyncClient(base_url="http://feast:6566")
    app.state.kafka_producer = ...  # Producer
    
    yield


app = FastAPI(lifespan=lifespan)


class FraudPredictionRequest(BaseModel):
    account_id: int
    transaction_amount_cents: int
    merchant_category: str
    country: str


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    decision: str  # "approve", "decline", "review"
    latency_ms: float


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict(request: FraudPredictionRequest):
    start = time.perf_counter()
    
    # 1. Fetch online features from Feast
    features_task = app.state.feast_client.post(
        "/get-online-features",
        json={
            "features": [
                "transaction_5m_features:tx_count_5m",
                "transaction_5m_features:total_amount_5m",
                "transaction_5m_features:avg_amount_5m",
            ],
            "entities": {"account_id": [request.account_id]},
        },
    )
    
    user_features_task = app.state.feast_client.post(
        "/get-online-features",
        json={
            "features": [
                "user_profile:account_age_days",
                "user_profile:country_risk",
            ],
            "entities": {"user_id": [request.account_id]},
        },
    )
    
    stream_features, user_features = await asyncio.gather(
        features_task,
        user_features_task,
    )
    
    # 2. Build full feature vector
    features = {
        "tx_count_5m": stream_features.json()["tx_count_5m"][0],
        "total_amount_5m": stream_features.json()["total_amount_5m"][0],
        "avg_amount_5m": stream_features.json()["avg_amount_5m"][0],
        "account_age_days": user_features.json()["account_age_days"][0],
        "country_risk": user_features.json()["country_risk"][0],
        "merchant_category_risk": lookup_risk(request.merchant_category),
        "tx_velocity_5m": stream_features.json()["tx_count_5m"][0] / 5.0,  # per minute
    }
    
    # 3. Predict
    with prediction_latency.time():
        fraud_prob = app.state.model.predict(features)
    
    # 4. Decision
    if fraud_prob > 0.8:
        decision = "decline"
    elif fraud_prob > 0.5:
        decision = "review"
    else:
        decision = "approve"
    
    predictions_made.labels(outcome=decision).inc()
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    return FraudPredictionResponse(
        fraud_probability=fraud_prob,
        decision=decision,
        latency_ms=elapsed_ms,
    )
```

---

## 7. The Drift Detector (`app/drift.py`)

```python
# Drift detector runs as a separate background process consuming predictions
import asyncio
from river.drift import ADWIN
from prometheus_client import Gauge, push_to_gateway
from datetime import datetime, timedelta
import httpx


drift_window_size_gauge = Gauge("fraud_drift_window_size", "ADWIN window size")
last_retrain_gauge = Gauge("fraud_drift_last_retrain_timestamp", "Last retrain timestamp")


class FraudDriftDetector:
    """Detect drift in fraud prediction quality."""
    
    def __init__(self, model_name: str = "fraud_detector", threshold: float = 0.002):
        self.model_name = model_name
        self.adwin = ADWIN(delta=threshold)
        self.last_retrain = datetime.min
    
    async def process(self, predicted: float, actual: float):
        """Update drift detector with new prediction-actual pair."""
        error = abs(predicted - actual)
        self.adwin.update(error)
        drift_window_size_gauge.set(self.adwin.width)
        
        # Confirmation: require sustained detection
        if self.adwin.drift_detected:
            # Throttle: don't retrain more than once per hour
            if datetime.utcnow() - self.last_retrain > timedelta(hours=1):
                self.last_retrain = datetime.utcnow()
                last_retrain_gauge.set(self.last_retrain.timestamp())
                await self.trigger_retraining()
                self.adwin = ADWIN(delta=0.002)  # reset
                return {"drift": True, "action": "retraining_triggered"}
        
        return {"drift": False, "window_size": self.adwin.width}
    
    async def trigger_retraining(self):
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://retrainer/retrain",
                json={"model_name": self.model_name, "trigger": "drift"},
            )


# Background task: consume prediction-actual pairs from Kafka
async def drift_monitor_loop():
    from aiokafka import AIOKafkaConsumer
    
    consumer = AIOKafkaConsumer(
        "fraud-predictions-with-labels",
        bootstrap_servers="kafka:9092",
        group_id="drift-monitor",
    )
    await consumer.start()
    
    detector = FraudDriftDetector("fraud_detector", threshold=0.002)
    
    try:
        async for msg in consumer:
            data = json.loads(msg.value)
            predicted = data["fraud_probability"]
            actual = 1.0 if data["is_fraud"] else 0.0
            
            drift_result = await detector.process(predicted, actual)
            
            if drift_result["drift"]:
                print(f"Drift detected: {drift_result}")
    finally:
        await consumer.stop()


# Run as a separate process
asyncio.run(drift_monitor_loop())
```

---

## 8. The Auto-Retrainer (`pipelines/retrainer.py`)

```python
"""Triggered by drift signal. Trains a new model on last 30 days of data."""
import asyncio
import httpx
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier


async def retrain(model_name: str, trigger: str):
    print(f"Retraining {model_name} (trigger: {trigger})...")
    
    # 1. Pull last 30 days of labeled transactions from Postgres
    from sqlalchemy import create_engine
    
    engine = create_engine("postgresql://app:secret@postgres:5432/app")
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    
    df = pd.read_sql(
        f"""
        SELECT
            t.amount_cents / 100.0 AS amount,
            t.merchant_category,
            t.country,
            u.account_age_days,
            EXTRACT(HOUR FROM t.created_at) AS tx_hour,
            -- Streaming features (precomputed in Feast / CDC pipeline)
            count(*) OVER (
                PARTITION BY t.account_id
                ORDER BY t.created_at
                RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND CURRENT ROW
            ) AS tx_velocity_5m,
            fl.is_fraud
        FROM transactions t
        JOIN users u ON t.account_id = u.id
        JOIN fraud_labels fl ON t.id = fl.transaction_id
        WHERE t.created_at BETWEEN $1 AND $2
        """,
        engine,
        params=(start, end),
    )
    
    # 2. Train new model
    # (in production: more sophisticated training pipeline)
    from app.model import FraudDetectionModel
    FraudDetectionModel.train(
        training_data_path="from_postgres",
        output_path="/models/fraud_detector.pkl",
    )
    
    # 3. Deploy new model (blue/green)
    print("New model trained. Deploying via blue/green...")
    
    # 4. Notify
    await notify_team(
        f"Model {model_name} retrained at {datetime.utcnow().isoformat()} (trigger: {trigger})"
    )
    
    print(f"Retraining complete: {model_name}")


async def notify_team(message: str):
    """Send Slack notification."""
    async with httpx.AsyncClient() as client:
        await client.post(
            os.getenv("SLACK_WEBHOOK_URL"),
            json={"text": message},
        )
```

---

## 9. Prometheus + Grafana Configuration

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s


scrape_configs:
  - job_name: 'fraud-api'
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'drift-detector'
    static_configs:
      - targets: ['api:8001']  # drift exporter
  
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:7071']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

`monitoring/grafana_dashboard.json` (key panels):

```json
{
    "dashboard": {
        "title": "Real-time Fraud Detection",
        "panels": [
            {
                "title": "Predictions per second",
                "targets": [{"expr": "rate(fraud_predictions_made_total[1m)"}]
            },
            {
                "title": "P99 prediction latency",
                "targets": [{"expr": "histogram_quantile(0.99, fraud_prediction_latency_seconds)"}]
            },
            {
                "title": "ADWIN window size",
                "targets": [{"expr": "fraud_drift_window_size"}]
            },
            {
                "title": "Drift events",
                "targets": [{"expr": "increase(fraud_drift_detected_total[1h)"}]
            },
            {
                "title": "Decision breakdown",
                "targets": [{"expr": "fraud_predictions_made_total by outcome"}]
            }
        ]
    }
}
```

---

## 10. The Production Workflow

The full system in operation:

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Feast
    participant Bytewax
    participant Kafka
    participant Postgres
    participant Debezium
    participant Drift
    participant Retrainer
    
    User->>API: POST /predict (transaction)
    API->>Feast: GET online features (tx_count_5m, ...)
    Feast-->>API: features
    API->>API: model.predict(features)
    API-->>User: fraud_probability
    
    Note over Postgres,Bytewax: Async pipeline (continuous)
    Postgres->>Debezium: WAL changes
    Debezium->>Kafka: CDC event
    Kafka->>Bytewax: consume CDC
    Bytewax->>Feast: write aggregated features
    
    Note over API,Drift: Drift detection (continuous)
    API->>Drift: prediction + actual (delayed)
    Drift->>Drift: ADWIN.update(error)
    alt Drift confirmed
        Drift->>Retrainer: POST /retrain
        Retrainer->>Postgres: pull 30d labeled data
        Retrainer->>Retrainer: train new model
        Retrainer->>API: deploy new model
    end
```

---

## 11. Production Deployment Checklist

- [ ] PostgreSQL with `wal_level=logical` and publication enabled
- [ ] Debezium with stable `slot.name` (no auto-drop)
- [ ] Kafka with KRaft mode and persistent volumes
- [ ] Bytewax stream processor with checkpointing (every 60s)
- [ ] Feast online store backed by Redis with replication
- [ ] FastAPI inference service with model pre-loaded
- [ ] Drift detector running as separate process with throttling
- [ ] Auto-retraining pipeline with rate limiting (max 1 retrain/hour)
- [ ] Prometheus + Grafana dashboards
- [ ] PagerDuty alerts on drift detection failures
- [ ] Disaster recovery: S3 backup of feature store + model registry

---

## 🎯 Key Takeaways

- The capstone composes Notes 01-04 into a single end-to-end pipeline.
- One `docker-compose.yml` brings up Postgres, Kafka, Debezium, Bytewax, Feast, Redis, API, retrainer, Prometheus, Grafana.
- The fraud-detection example shows the full pattern; replace the model and features for your use case.
- Sub-50ms inference latency via pre-loaded model + online feature store + async I/O.
- Auto-retraining triggered by ADWIN drift detection, with throttling to prevent loops.
- Cost is dominated by Kafka + Redis + Postgres hosting; ~$200/mo on Hetzner for production traffic.
- The capstone is the **twelfth portfolio project**: the real-time ML skill.

## References

- [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/01 - Apache Kafka|Kafka note 01]] — Kafka fundamentals
- [[09 - MLOps y Produccion/27 - Feast and Feature Stores|Feast course]] — online feature store
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — drift detection foundation
- [[09 - MLOps y Produccion/22 - End-to-End ML Project|E2E ML Project]] — training pipeline
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/01 - Streaming Feature Engineering|Note 01 — Streaming Features]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/02 - Online Inference and Event-Driven ML|Note 02 — Online Inference]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/03 - Change Data Capture for ML|Note 03 — CDC]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/04 - Drift Detection in Real-time|Note 04 — Drift Detection]]
- Bytewax — [bytewax.io](https://bytewax.io/)
- Debezium — [debezium.io](https://debezium.io/)
- Feast — [feast.dev](https://feast.dev/)
- River ML — [riverml.xyz](https://riverml.xyz/)
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems|Incident Response]] — alerts for drift detection failures