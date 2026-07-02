# Ray for Distributed ML

## 1. Ray Core: Tasks and Actors

Ray transforms Python functions into remote tasks and classes into remote actors.

### Tasks

```python
import ray

ray.init()

@ray.remote
def preprocess_chunk(data: list[float]) -> list[float]:
    return [x * 2 for x in data]

futures = [preprocess_chunk.remote(chunk) for chunk in chunks]
results = ray.get(futures)  # block until all complete
```

### Actors

```python
@ray.remote
class ModelReplica:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, input: dict) -> dict:
        return self.model.predict(input)

replica = ModelReplica.remote("model.pt")
result = ray.get(replica.predict.remote({"features": [1.0, 2.0]}))
```

Actors maintain state across calls. Use for model replicas that hold GPU memory.

## 2. Ray Serve: Model Deployment

```python
from ray import serve
from starlette.requests import Request

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 10, "target_num_ongoing_requests_per_replica": 5},
)
class Predictor:
    def __init__(self):
        self.model = load_model("model.pt")

    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        return {"prediction": self.model.predict(data["features"]).tolist()}

serve.run(Predictor.bind())
```

### Canary Deployments

```python
@serve.deployment(version="v2", route_prefix="/predict")
class PredictorV2(Predictor):
    def __init__(self):
        self.model = load_model("model-v2.pt")

# Deploy and route 10% traffic
serve.run(PredictorV2.bind(), route_prefix="/predict")
```

### Model Composition

```python
@serve.deployment
class Router:
    def __init__(self):
        self.v1 = PredictorV1.get_handle()
        self.v2 = PredictorV2.get_handle()

    async def __call__(self, request):
        if request.headers.get("X-Model-Version") == "v2":
            return await self.v2.remote(request)
        return await self.v1.remote(request)
```

## 3. Ray Train: Distributed Training

```python
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    model = create_model()
    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = train_step(model, batch)
            print(f"epoch {epoch}, loss {loss}")

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(storage_path="s3://bucket/checkpoints"),
)
result = trainer.fit()
```

## 4. Ray Data: Dataset Processing

```python
import ray

ds = ray.data.read_parquet("s3://bucket/data/")
ds = ds.map_batches(
    preprocess_chunk,
    batch_size=128,
    num_cpus=2,
)
ds.write_parquet("s3://bucket/processed/")
```

For online inference preprocessing:

```python
@serve.deployment
class Preprocessor:
    async def __call__(self, request: Request):
        data = await request.json()
        # Use Ray Data for batch preprocessing within the request
        ds = ray.data.from_items(data["items"])
        processed = ds.map_batches(preprocess_fn).take_all()
        return {"processed": processed}
```

## 5. Ray Cluster Setup

```python
# On head node
ray start --head --port=6379 --num-gpus=8

# On worker nodes
ray start --address=<head-ip>:6379 --num-gpus=8

# Or with Kubernetes
```

Kubernetes:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ml-cluster
spec:
  workerGroupSpecs:
  - groupName: gpu-group
    replicas: 3
    rayActorOptions:
      numGPUs: 1
```

## 6. Ray + FastAPI Integration

```python
from ray import serve
from fastapi import FastAPI

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MLService:
    def __init__(self):
        self.model = load_model()

    @app.post("/predict")
    async def predict(self, data: dict):
        return {"prediction": self.model.predict(data["features"]).tolist()}
```

## Key Takeaways

- **Ray Core**: `@ray.remote` for tasks (stateless) and actors (stateful)
- **Ray Serve**: deploy models with autoscaling, canary, composition
- **Ray Train**: distributed training with `TorchTrainer` + `ScalingConfig`
- **Ray Data**: parallel dataset processing with `map_batches`
- **Ray Cluster**: `ray start` on nodes or K8s `RayCluster` custom resource
- **Integration**: `@serve.ingress(app)` combines FastAPI + Ray
