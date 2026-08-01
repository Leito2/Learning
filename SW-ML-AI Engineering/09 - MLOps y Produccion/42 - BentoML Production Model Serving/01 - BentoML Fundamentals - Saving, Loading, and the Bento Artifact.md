# 🎯 01 - BentoML Fundamentals — Saving, Loading, and the Bento Artifact

> **The patterns. Save any model as a Bento, load it back, version it, and serve it. The foundation of every BentoML service.**

## 🎯 Learning Objectives
- Save models from sklearn, PyTorch, XGBoost, TensorFlow, HuggingFace
- Understand the Bento artifact (model + code + environment)
- Version models and load specific versions
- Build a basic service with `@bentoml.service` and `@bentoml.api`
- Run the Bento locally with `bentoml serve`
- Build a Docker image with `bentoml build`
- Use the OpenAPI/Swagger UI for free

## Introduction

A Bento is a **standardized artifact** that contains:

1. **The model** — serialized weights (sklearn, PyTorch, etc.)
2. **The code** — prediction logic
3. **The environment** — Python version, dependencies (conda/pip)
4. **The metadata** — version, name, framework, custom tags

This is the equivalent of a Docker image but for ML models. Build once, deploy anywhere.

```bash
# Save a model
bentoml.sklearn.save_model("my_model", model)

# List versions
bentoml models list my_model

# Build a Bento (container)
bentoml build

# Serve locally
bentoml serve

# Deploy to K8s
bentoml deploy --platform kubernetes
```

This note covers the model saving/loading and the basic service.

---

## 1. Your First Bento

```python
# train.py
import bentoml
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train
model = svm.SVC()
model.fit(X_train, y_train)

# Save as Bento
bentoml.sklearn.save_model(
    "iris_classifier",
    model,
    signatures={"predict": {"batchable": True, "batch_dim": 0}},
    metadata={"framework": "sklearn", "accuracy": 0.97},
    labels={"env": "dev", "team": "data-science"},
)
```

```bash
# View saved models
bentoml models list
# Tag: latest, dev:iris_classifier:l3kxj2...

# Get model info
bentoml models get iris_classifier:latest
```

The model is saved with multi-framework support: `bentoml.sklearn`, `bentoml.pytorch`, `bentoml.tensorflow`, `bentoml.xgboost`, `bentoml.huggingface`.

---

## 2. The Bento Artifact

A Bento is a directory that contains:

```
my_bento/
├── bento.yaml             # Bento metadata
├── apis/                  # Service code
│   └── service.py
├── src/                   # Custom code
├── models/                # Model weights
│   └── iris_classifier/
│       ├── model.pkl
│       └── model.yaml
├── env/                   # Python environment
│   └── environment.yml
└── requirements.txt
```

The `bento.yaml`:
```yaml
service: "service:IrisService"
labels:
  owner: data-science
  project: iris
models:
  - "iris_classifier:l3kxj2..."
include:
  - "**/*.py"
  - "models/*.pkl"
exclude:
  - "**/__pycache__/"
```

The Bento is **portable** — same content on Docker, Kubernetes, Lambda, your laptop.

---

## 3. Loading a Model

```python
# In your service
import bentoml

# Load latest version
model = bentoml.sklearn.load_model("iris_classifier:latest")

# Load specific version
model = bentoml.sklearn.load_model("iris_classifier:dev")

# Load by tag
model = bentoml.sklearn.load_model("iris_classifier:production")
```

For multi-model services:

```python
classifier = bentoml.sklearn.load_model("iris_classifier:latest")
regressor = bentoml.sklearn.load_model("price_predictor:latest")
```

---

## 4. The Service Class

```python
# service.py
import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel


class FeatureInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionOutput(BaseModel):
    species: str
    confidence: float


iris_model_runner = bentoml.sklearn.load_model_runner("iris_classifier:latest")


@bentoml.service(
    name="iris_service",
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 30, "concurrency": 100},
)
class IrisService:
    model = bentoml.models.BentoModel("iris_classifier:latest")
    
    def __init__(self):
        import joblib
        self.model = joblib.load(self.model.path_of("model.pkl"))
    
    @bentoml.api(batchable=True, batch_dim=0)
    def predict(self, input_data: list[dict]) -> list[dict]:
        """Predict Iris species from flower measurements."""
        results = []
        for item in input_data:
            features = np.array([
                item["sepal_length"],
                item["sepal_width"],
                item["petal_length"],
                item["petal_width"],
            ]).reshape(1, -1)
            
            species_idx = self.model.predict(features)[0]
            confidence = float(self.model.predict_proba(features).max())
            
            results.append({
                "species": ["setosa", "versicolor", "virginica"][species_idx],
                "confidence": confidence,
            })
        
        return results
```

The service:
- Loads the model at startup
- Defines a REST endpoint at `/predict`
- Type-safe via Pydantic
- **Batchable** — multiple requests in one batch

---

## 5. Serving Locally

```bash
# Serve in development mode
bentoml serve service:IrisService

# Access (auto-generated Swagger UI)
# http://localhost:3000/predict
# http://localhost:3000/docs.json (OpenAPI spec)
```

The Swagger UI is auto-generated from the type annotations. Free API docs.

```bash
# Test the API
curl -X POST http://localhost:3000/predict \
    -H "Content-Type: application/json" \
    -d '{"input_data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'
```

---

## 6. Building a Production Docker Image

```bash
# Build the Bento
bentoml build

# List built Bentos
bentoml list
# Tag: latest, dev:iris_service:abc123

# Build container image
bentoml containerize iris_service:latest
# → iris_service:abc123 OCI image
```

The container is a standard OCI image. Deploy anywhere.

```bash
# Run the container
docker run -p 3000:3000 iris_service:abc123

# Or push to a registry
docker tag iris_service:abc123 my-registry.com/iris_service:1.0
docker push my-registry.com/iris_service:1.0
```

---

## 7. Multi-Framework Support

### 7.1 PyTorch

```python
import torch
import bentoml


class IrisNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)
    
    def forward(self, x):
        return self.fc(x)


model = IrisNet()
model.load_state_dict(torch.load("model.pt"))

# Save as Bento
bentoml.pytorch.save_model(
    "iris_nn",
    model,
    signatures={"predict": {"batchable": True}},
)
```

### 7.2 XGBoost

```python
import xgboost as xgb
import bentoml

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

bentoml.xgboost.save_model(
    "iris_xgb",
    model,
    signatures={"predict": {"batchable": True}},
)
```

### 7.3 HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import bentoml

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

bentoml.transformers.save_model(
    "gpt2",
    model,
    signatures={"generate": {"batchable": False}},
    metadata={"tokenizer": tokenizer},
)
```

### 7.4 Custom model

```python
import bentoml
import numpy as np


class MyCustomModel:
    def predict(self, X):
        return np.zeros(len(X))


# Save any Python object with pickle
bentoml.picklable_model.save_model(
    "my_custom_model",
    MyCustomModel(),
    signatures={"predict": {"batchable": False}},
)
```

---

## 8. Multi-Model Services

For services that use multiple models:

```python
@bentoml.service
class MultiModelService:
    # Multiple models
    encoder = bentoml.models.BentoModel("text_encoder:latest")
    classifier = bentoml.models.BentoModel("classifier:latest")
    
    def __init__(self):
        # Load at startup
        from transformers import AutoModel
        self.encoder_model = AutoModel.from_pretrained(self.encoder.path)
        # Load classifier
        self.clf = bentoml.sklearn.load_model(self.classifier)
    
    @bentoml.api
    def predict(self, text: str) -> dict:
        # Encode
        embedding = self.encoder_model.encode(text)
        
        # Classify
        prediction = self.clf.predict([embedding])
        
        return {"label": prediction[0], "confidence": 0.95}
```

Multi-model services are common for:
- Embeddings + downstream classifiers
- Pre-processing + main model + post-processing
- Ensembles (multiple models voting)

---

## 9. Versioning and Model Registry

```bash
# Log a new version
bentoml.sklearn.save_model("iris_classifier", model_v2)

# Browse versions
bentoml models list iris_classifier
# Tag: latest, dev:iris_classifier:l3kxj2...
# Tag: prod:iris_classifier:abc123

# Get details
bentoml models get iris_classifier:prod

# Delete a version
bentoml models delete iris_classifier:dev

# Export a Bento (for sharing)
bentoml export iris_service:latest ./iris_service.bento
```

The model registry is the **source of truth** for which model version is deployed.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Loading models in the request handler

```python
# ❌ Slow: loads model every request
@bentoml.api
def predict(self, input_data: list[dict]) -> list[dict]:
    model = bentoml.sklearn.load_model("iris_classifier:latest")  # slow!
    return model.predict(input_data).tolist()

# ✅ Load once at startup
@bentoml.service
class IrisService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("iris_classifier:latest")
    
    @bentoml.api
    def predict(self, input_data):
        return self.model.predict(input_data).tolist()
```

### 10.2 Antipattern 2: Not using the model runner

```python
# ❌ Loads model in service (manual GPU management)
@bentoml.service
class MyService:
    def __init__(self):
        self.model = bentoml.pytorch.load_model("my_model")
        self.model.cuda()  # manual GPU placement

# ✅ Use the runner pattern (handles GPU, batching, async)
model_runner = bentoml.pytorch.load_model_runner("my_model")

@bentoml.service
class MyService:
    model = bentoml.models.BentoModel("my_model:latest")
    
    def __init__(self):
        self.model_runner = bentoml.pytorch.load_model_runner(self.model)
```

### 10.3 Antipattern 3: Storing PII in the model metadata

```python
# ❌ PII in metadata
bentoml.sklearn.save_model(
    "model",
    model,
    metadata={"user_email": "alice@example.com"},  # PII!
)

# ✅ Anonymized metadata
bentoml.sklearn.save_model(
    "model",
    model,
    metadata={"user_id": "user-123", "training_date": "2026-07-23"},
)
```

### 10.4 Antipattern 4: No input validation

```python
# ❌ Trusts input
@bentoml.api
def predict(self, input_data):
    return self.model.predict(input_data)

# ✅ Validate input
from pydantic import BaseModel, Field

class PredictInput(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)

@bentoml.api
def predict(self, input: PredictInput) -> dict:
    return {"prediction": self.model.predict([input.features]).tolist()}
```

### 10.5 Antipattern 5: Not versioning for rollback

```python
# ❌ Single version, no rollback
bentoml.sklearn.save_model("iris_classifier", model_v2)  # overwrites latest

# ✅ Stack versions for rollback
bentoml.sklearn.save_model("iris_classifier", model_v2)  # new version
# Keep old: dev:iris_classifier:abc123
# Rollback: load_model("iris_classifier:abc123")
```

---

## 🎯 Key Takeaways

- BentoML saves any model as a standardized Bento (model + code + environment).
- Multi-framework: sklearn, PyTorch, XGBoost, TensorFlow, HuggingFace, custom.
- The service class with `@bentoml.service` and `@bentoml.api` is the production pattern.
- `bentoml serve` for local dev; `bentoml containerize` for Docker.
- Auto-generated OpenAPI/Swagger UI for free API docs.
- Multi-model services for embeddings + classifiers.
- Versioning with tags; rollback is `load_model("model:old_tag")`.
- Avoid loading models in handlers, manual GPU, PII in metadata, no validation, no version rollback.

## References

- BentoML docs — [docs.bentoml.com](https://docs.bentoml.com)
- BentoML GitHub — [github.com/bentoml/BentoML](https://github.com/bentoml/BentoML)
- BentoML examples — [github.com/bentoml/BentoML/tree/main/examples](https://github.com/bentoml/BentoML/tree/main/examples)
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM]] — LLM serving comparison
- [[09 - MLOps y Produccion/32 - KServe and Knative\|KServe]] — K8s-native serving
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] — custom FastAPI alternative
- [[09 - MLOps y Produccion/42 - BentoML Production Model Serving/02 - Service Patterns - REST, gRPC, Async, Batching|Note 02 — Service Patterns]]