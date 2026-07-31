# 🎯 04 - Drift Detection in Real-time — ADWIN, Page-Hinkley, and Auto-Retraining

> **Detect when your model's quality degrades. In real time. Triggered on streaming data, not nightly batches. The pattern that keeps production ML systems honest.**

## 🎯 Learning Objectives
- Distinguish concept drift, data drift, and label drift
- Implement streaming drift detection with ADWIN, Page-Hinkley, EDDM
- Set up real-time drift dashboards with Evidently + Prometheus
- Configure auto-retraining triggered by drift signals
- Use the drift detection patterns from [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently]] in production
- Calibrate drift thresholds to minimize false positives
- Apply drift detection to portfolio projects (LLM Eval Suite, AutoTrain)

## Introduction

Models degrade in production. Data distributions shift. User behavior changes. Adversaries adapt. Without drift detection, your model silently becomes wrong.

**Evidently AI** (covered in [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|09/31]]) gives you offline drift reports on batch data. But real-time drift detection requires **streaming algorithms**: ADWIN, Page-Hinkley, EDDM. These process one sample at a time, in O(1) memory, and detect distribution shifts as they happen.

This note covers:
- The three types of drift (concept, data, label)
- The five streaming drift algorithms (ADWIN, Page-Hinkley, EDDM, DDM, CUSUM)
- Real-time drift dashboards (Evidently + Prometheus + Grafana)
- Auto-retraining pipelines triggered by drift
- Calibration to minimize false positives

![Drift detection pipeline](https://example.com/drift-detection.png)

---

## 1. The Three Types of Drift

| Type | What changes | Example |
|------|-------------|---------|
| **Concept drift** | The relationship between features and labels changes | User preferences shift from "fast delivery" to "low price" |
| **Data drift** | The feature distribution changes | New device type appears in production traffic |
| **Label drift** | The label distribution changes (often rare in real-time) | Class imbalance shifts due to seasonality |

Concept drift is the most common in real-time ML. The model's learned relationship is now stale.

---

## 2. The Five Streaming Drift Algorithms

### 2.1 ADWIN (Adaptive Windowing)

ADWIN maintains a variable-length window and detects changes by comparing sub-windows.

```python
from river.drift import ADWIN


adwin = ADWIN(delta=0.002)  # confidence threshold (smaller = more sensitive)


def process_prediction(model_output, actual_label):
    """Update ADWIN with each new labeled sample."""
    error = 1 if (model_output > 0.5) != (actual_label > 0.5) else 0
    adwin.update(error)
    
    if adwin.drift_detected:
        return {
            "drift_detected": True,
            "window_size": adwin.width,
            "action": "trigger_retraining",
        }
    return {"drift_detected": False, "window_size": adwin.width}
```

**ADWIN is the default for real-time drift detection**. Memory-efficient, theoretically grounded, widely used.

### 2.2 Page-Hinkley

Detects changes in the mean of a sequence. Faster than ADWIN but less robust to noise.

```python
from river.drift import PageHinkley


ph = PageHinkley(min_instances=30, threshold=50, alpha=0.9999)


def process_prediction(model_output, actual_label):
    error = abs(model_output - actual_label)
    ph.update(error)
    
    if ph.drift_detected:
        return {"drift_detected": True, "action": "trigger_retraining"}
    return {"drift_detected": False}
```

Page-Hinkley is faster to detect change but has more false positives. Use when you need immediate detection.

### 2.3 EDDM (Early Drift Detection Method)

Detects gradual drift. Best for cases where the distribution shifts slowly.

```python
from river.drift import EDDM


eddm = EDDM()


def process_prediction(error_distance):
    eddm.update(error_distance)
    if eddm.drift_detected:
        return {"drift_detected": True, "action": "trigger_retraining"}
    return {"drift_detected": False}
```

EDDM works by monitoring the distance between errors. When the average distance grows, drift is suspected.

### 2.4 DDM (Drift Detection Method)

The original Page-Hinkley variant. Best for sudden, dramatic changes.

```python
from river.drift import DDM


ddm = DDM(min_num_instances=30, warning_threshold=2.0, drift_threshold=3.0)
```

DDM has two thresholds:
- `warning_threshold` — flag the sample but don't trigger retraining
- `drift_threshold` — trigger retraining

### 2.5 CUSUM (Cumulative Sum)

Detects small, sustained shifts. Useful for monitoring metrics that should stay constant (e.g., latency, cost).

```python
from river.drift import CUSUM


cusum_latency = CUSUM(threshold=100, drift_threshold=200)


def process_latency(latency_ms):
    cusum_latency.update(latency_ms)
    if cusum_latency.drift_detected:
        return {"latency_drift": True, "action": "investigate"}
    return {"latency_drift": False}
```

---

## 3. Algorithm Selection

| Algorithm | Best for | Strengths | Weaknesses |
|-----------|----------|-----------|-----------|
| **ADWIN** | General concept drift | Memory-efficient, theoretically grounded | Slower for sudden changes |
| **Page-Hinkley** | Latency/cost drift | Fast detection | More false positives |
| **EDDM** | Gradual drift | Detects slow shifts | Misses sudden shifts |
| **DDM** | Sudden drift | Simple, fast | Requires labeled data quickly |
| **CUSUM** | Operational metrics | Detects sustained small shifts | Needs calibration |

For most real-time ML systems, **ADWIN is the default**. Combine with **CUSUM** for operational metrics (latency, cost, error rate).

---

## 4. Real-time Drift Detection Pipeline

```python
# drift_detector.py
import asyncio
from river.drift import ADWIN
from prometheus_client import Gauge, Counter


# Prometheus metrics
drift_detected = Gauge(
    "model_drift_detected",
    "1 if drift detected, 0 otherwise",
)
drift_window_size = Gauge(
    "model_drift_window_size",
    "Current ADWIN window size",
)
drift_retraining_triggered = Counter(
    "model_drift_retraining_triggered_total",
    "Number of times drift triggered retraining",
)


class DriftDetector:
    """Detect drift in real time, trigger retraining."""
    
    def __init__(self, model_name: str, drift_threshold: float = 0.002):
        self.model_name = model_name
        self.adwin = ADWIN(delta=drift_threshold)
        self.recent_predictions: deque = deque(maxlen=1000)
    
    async def process(self, prediction: float, actual: float | None = None):
        """Process a prediction (and optional actual label)."""
        
        # Track prediction
        self.recent_predictions.append({
            "timestamp": datetime.utcnow(),
            "prediction": prediction,
            "actual": actual,
        })
        
        # Skip if no actual label yet
        if actual is None:
            return {"drift_detected": False, "reason": "no_label_yet"}
        
        # Compute error
        error = abs(prediction - actual)
        
        # Update ADWIN
        self.adwin.update(error)
        
        # Update metrics
        drift_window_size.labels(model=self.model_name).set(self.adwin.width)
        
        # Check for drift
        if self.adwin.drift_detected:
            drift_detected.labels(model=self.model_name).set(1)
            drift_retraining_triggered.labels(model=self.model_name).inc()
            
            # Trigger retraining
            await self.trigger_retraining()
            
            # Reset ADWIN after drift is handled
            self.adwin = ADWIN(delta=0.002)
            
            return {
                "drift_detected": True,
                "window_size": self.adwin.width,
                "action": "retraining_triggered",
            }
        
        drift_detected.labels(model=self.model_name).set(0)
        return {"drift_detected": False, "window_size": self.adwin.width}
    
    async def trigger_retraining(self):
        """Trigger the retraining pipeline."""
        # Call the training service
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://training-service/retrain",
                params={"model_name": self.model_name, "trigger": "drift"},
            )
```

---

## 5. Wiring Drift Detection to LangFuse

LangFuse (covered in [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|09/36]]) tracks prediction quality via LLM-as-judge scores. Combine with ADWIN:

```python
from langfuse import observe


@observe()
async def process_prediction_with_drift(
    request: PredictionRequest,
    drift_detector: DriftDetector,
):
    """Make a prediction and check for drift."""
    
    # 1. Get prediction (covered in Note 02)
    prediction = await inference_service.predict(request.features)
    
    # 2. Get actual label (from delayed feedback)
    actual_label = await get_actual_label_async(request, delay_seconds=300)
    
    # 3. Update drift detector
    drift_result = await drift_detector.process(
        prediction=prediction["score"],
        actual=actual_label,
    )
    
    if drift_result["drift_detected"]:
        # LangFuse automatically captures this as an event
        langfuse_context.update_current_observation(
            metadata={"drift_detected": True, "drift_window": drift_result["window_size"]},
        )
    
        # Trigger retraining (handled by drift_detector.trigger_retraining)
        # ...
    
    return prediction
```

The full chain:
1. Inference service produces predictions
2. Actual labels arrive (delayed feedback, human review, or holdout set)
3. Drift detector compares predicted vs actual
4. If drift detected → trigger retraining
5. New model deployed
6. Drift detector resets

---

## 6. Drift Dashboards with Evidently + Prometheus

```python
# drift_dashboard.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def generate_drift_report(reference_data, current_data, output_path):
    """Generate an Evidently drift report."""
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)


def export_drift_to_prometheus(drift_results):
    """Export Evidently drift results to Prometheus."""
    from prometheus_client import Gauge, push_to_gateway
    
    drifted_features = Gauge(
        "evidently_drifted_features",
        "Number of features with detected drift",
        ["model"],
    )
    drift_score = Gauge(
        "evidently_drift_score",
        "Overall drift score",
        ["model"],
    )
    
    drifted_features.labels(model="CreditScoringLLM").set(
        drift_results["number_of_drifted_columns"]
    )
    drift_score.labels(model="CreditScoringLLM").set(
        drift_results["dataset_drift"]
    )
    
    push_to_gateway(
        gateway="http://prometheus-pushgateway:9091",
        job="drift_metrics",
        grouping_key={"model": "CreditScoringLLM"},
    )
```

The Grafana dashboard shows:
- Number of drifted features
- Overall drift score over time
- Per-feature drift scores
- ADWIN window size
- Retraining triggers

---

## 7. Auto-Retraining Pipeline

The auto-retraining pipeline is triggered by drift:

```python
# retraining_pipeline.py
import asyncio
import httpx
from datetime import datetime


class AutoRetrainingPipeline:
    """Trigger retraining on drift signals."""
    
    def __init__(self, training_service_url: str, model_name: str):
        self.training_service_url = training_service_url
        self.model_name = model_name
        self.last_retrain = datetime.min
    
    async def on_drift_detected(self, drift_result: dict):
        """Handle a drift detection event."""
        
        # Throttle: don't retrain more than once per hour
        if (datetime.utcnow() - self.last_retrain).seconds < 3600:
            print(f"Throttled: last retrain was {self.last_retrain}")
            return
        
        self.last_retrain = datetime.utcnow()
        
        # Trigger retraining
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.training_service_url}/retrain",
                json={
                    "model_name": self.model_name,
                    "trigger": "drift",
                    "drift_details": drift_result,
                    "training_data_window": "30d",
                },
                timeout=300.0,
            )
            
            if response.status_code == 200:
                print(f"Retraining triggered: {response.json()}")
            else:
                print(f"Retraining failed: {response.text}")


# Wire to drift detector
async def setup_drift_and_retraining():
    drift_detector = DriftDetector("CreditScoringLLM", drift_threshold=0.002)
    retraining = AutoRetrainingPipeline("http://training-service", "CreditScoringLLM")
    
    async def on_drift(change_event):
        drift_result = await drift_detector.process(...)
        if drift_result["drift_detected"]:
            await retraining.on_drift_detected(drift_result)
    
    # Wire to CDC consumer
    for change_event in consume_cdc_events("cdc.public.users"):
        await on_drift(change_event)
```

The end-to-end pipeline:
1. Drift detector runs on every labeled prediction
2. If drift detected → trigger retraining pipeline
3. Retraining trains new model on last 30 days of data
4. New model deployed via blue/green
5. Drift detector resets

---

## 8. Calibration and False Positives

False positives are the biggest operational risk. A drift detector that triggers retraining every 6 hours is useless.

```python
# Calibrate drift threshold using labeled validation data
def calibrate_drift_threshold(historical_predictions, historical_labels):
    """Find drift threshold that minimizes false positives."""
    
    best_threshold = 0.002
    best_f1 = 0
    
    for threshold in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
        detector = ADWIN(delta=threshold)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Simulate drift scenarios
        for pred, actual in zip(historical_predictions, historical_labels):
            detector.update(abs(pred - actual))
            # ... compute TP/FP/TN/FN ...
        
        # Compute F1
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
```

Typical production settings:
- ADWIN delta: 0.001 - 0.005 (depending on noise tolerance)
- Minimum window size: 100-1000 samples
- Confirm drift with at least 2 consecutive detections

---

## 9. Case Study — LLM Eval Suite with Drift Detection

The Automated LLM Evaluation Suite (your portfolio project) can benefit from drift detection:

```python
# Drift detection on LLM quality scores
class LLMEvalDriftDetector:
    """Detect drift in LLM eval quality over time."""
    
    def __init__(self):
        self.adwin = ADWIN(delta=0.005)
        self.recent_scores = deque(maxlen=500)
    
    def process(self, score: float):
        """Update drift detector with new eval score."""
        self.recent_scores.append(score)
        
        # ADWIN tracks the deviation from expected distribution
        # If score distribution shifts (e.g., due to model update), ADWIN detects it
        expected_mean = 0.85  # baseline
        deviation = abs(score - expected_mean)
        
        self.adwin.update(deviation)
        
        if self.adwin.drift_detected:
            return {
                "drift_detected": True,
                "recent_mean": sum(self.recent_scores) / len(self.recent_scores),
                "action": "investigate_model_or_data",
            }
        return {"drift_detected": False}
```

If the LLM quality drops (mean from 0.85 to 0.75), the deviation increases, ADWIN detects drift, and the team gets notified.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Drift detection on every sample

```python
# ❌ Drift detection on every sample — noisy
for sample in stream:
    if adwin.update(sample):
        trigger_retraining()  # fires on noise

# ✅ Confirm with sustained detection
class ConfirmationDriftDetector:
    def __init__(self, confirm_count=3):
        self.adwin = ADWIN(delta=0.002)
        self.consecutive_detections = 0
        self.confirm_count = confirm_count
    
    def process(self, sample):
        self.adwin.update(sample)
        if self.adwin.drift_detected:
            self.consecutive_detections += 1
            if self.consecutive_detections >= self.confirm_count:
                return {"drift": True, "confirmed": True}
        else:
            self.consecutive_detections = 0
        return {"drift": False}
```

### 10.2 Antipattern 2: No labeled data for concept drift

```python
# ❌ Concept drift detection needs labels; this is data drift
adwin.update(feature_value)  # tracks input distribution, not label

# ✅ Concept drift requires predicted vs actual
adwin.update(abs(predicted - actual))
```

Concept drift (model quality changing) requires ground truth labels. Data drift (input distribution changing) doesn't.

### 10.3 Antipattern 3: Single drift threshold

```python
# ❌ Single threshold for all metric types
threshold = 0.002  # same for latency, accuracy, cost

# ✅ Per-metric thresholds
THRESHOLDS = {
    "accuracy": 0.005,
    "latency_ms": 50,
    "cost_per_call": 0.001,
}
```

### 10.4 Antipattern 4: No throttling on retraining

```python
# ❌ Trigger retraining on every drift
if drift_detected:
    trigger_retraining()  # fires 10x per day

# ✅ Throttle to max once per hour or per day
last_retrain = None
if drift_detected and (now - last_retrain) > timedelta(hours=1):
    trigger_retraining()
    last_retrain = now
```

### 10.5 Antipattern 5: Drift detection without action

```python
# ❌ Detect drift, log it, but do nothing
if adwin.drift_detected:
    log.warning("Drift detected")

# ✅ Detect drift, trigger action (retrain, alert, disable feature)
if adwin.drift_detected:
    trigger_retraining()
    alert_oncall()
    consider_disabling_feature()
```

---

## 🎯 Key Takeaways

- Three drift types: concept (model quality), data (input distribution), label (label distribution).
- Five streaming algorithms: ADWIN (default), Page-Hinkley (latency), EDDM (gradual), DDM (sudden), CUSUM (operational).
- ADWIN with delta=0.002 is the production default for real-time concept drift.
- Confirm drift with sustained detection (3+ consecutive) before triggering retraining.
- Auto-retraining pipeline: drift detector → trigger → train → deploy → reset detector.
- Calibrate thresholds using historical data to minimize false positives.
- Throttle retraining to max once per hour; manual approval for high-cost retrains.
- Wire drift detection to LangFuse for unified observability.
- Avoid drift detection on every sample, missing labels, single threshold, no throttling, drift without action.

## References

- River ML — [riverml.xyz](https://riverml.xyz/)
- ADWIN paper — [riverml.xyz/latest/api/drift/ADWIN](https://riverml.xyz/latest/api/drift/ADWIN/)
- Evidently AI — [evidentlyai.com](https://evidentlyai.com/)
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently course]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]]
- [[09 - MLOps y Produccion/22 - End-to-End ML Project|E2E ML Project]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/01 - Streaming Feature Engineering|Note 01 — Streaming Features]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/02 - Online Inference and Event-Driven ML|Note 02 — Online Inference]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/05 - Capstone - Production Real-time ML Pipeline|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems|Incident Response]] — drift triggers alerts