# 🎯 04 - Deployment Targets — Kubernetes, Lambda, SageMaker, Cloud Run

> **Build once, deploy anywhere. BentoML's strength is portable deployments. The patterns for each cloud and on-prem target.**

## 🎯 Learning Objectives
- Deploy to Kubernetes with Helm + auto-scaling
- Deploy to AWS Lambda for serverless ML
- Deploy to AWS SageMaker for managed inference
- Deploy to Google Cloud Run for serverless containers
- Deploy to Azure ML for enterprise hybrid
- Self-host with Docker Compose for local dev
- Choose the right deployment target for your workload

## Introduction

BentoML's killer feature is **portable deployment**. The same Bento deploys to:

| Target | Use case |
|--------|----------|
| **Kubernetes** | Production scale, full control |
| **AWS Lambda** | Bursty traffic, pay-per-request |
| **AWS SageMaker** | AWS-native, managed inference |
| **Google Cloud Run** | Serverless containers |
| **Azure ML** | Enterprise hybrid |
| **Docker Compose** | Local dev, simple deploys |

This note covers each target with the right configuration and the tradeoffs.

---

## 1. Kubernetes — Production Scale

```bash
# Build the Bento
bentoml build

# Deploy with K8s (uses Helm under the hood)
bentoml deploy iris_service:latest --platform kubernetes \
    --replicas 3 \
    --cpu 2 \
    --memory 4Gi \
    --gpu 0 \
    --namespace production \
    --ingress-enabled
```

This creates:
- Deployment with 3 replicas
- Service exposing the API
- Ingress with TLS
- HPA (Horizontal Pod Autoscaler) auto-configured

### 1.1 The Generated Kubernetes Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-service
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-service
  template:
    metadata:
      labels:
        app: iris-service
    spec:
      containers:
        - name: iris-service
          image: iris_service:latest
          ports:
            - containerPort: 3000
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          readinessProbe:
            httpGet:
              path: /healthz
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: 3000
            initialDelaySeconds: 30
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: iris-service
spec:
  selector:
    app: iris-service
  ports:
    - port: 80
      targetPort: 3000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-service-hpa
spec:
  scaleTargetRef:
    name: iris-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 1.2 Custom K8s manifest

```yaml
# custom-deploy.yaml
apiVersion: serving.bentoml.org/v1
kind: BentoService
metadata:
  name: iris-service
spec:
  image: iris_service:latest
  replicas: 3
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  ingress:
    enabled: true
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
  serviceMonitor:
    enabled: true  # Prometheus metrics
  podDisruptionBudget:
    minAvailable: 2
```

Apply:
```bash
kubectl apply -f custom-deploy.yaml
```

---

## 2. AWS Lambda — Bursty Traffic

```bash
# Build and deploy to Lambda
bentoml deploy iris_service:latest --platform aws-lambda \
    --region us-east-1 \
    --memory 1024 \
    --timeout 30 \
    --api-gateway-enabled
```

This creates:
- Lambda function with the Bento image
- API Gateway endpoint
- IAM role with permissions

### 2.1 Lambda Constraints

| Constraint | Value |
|------------|-------|
| Max memory | 10 GB |
| Max execution time | 900s (15 min) |
| Max payload | 6 MB (sync), 20 MB (async) |
| Cold start | 1-3 seconds |

For ML inference:
- Small models (< 1 GB) work great
- Large models (> 1 GB) — use EFS for model storage
- Streaming — not supported; use async invocation

### 2.2 EFS for Large Models

```python
# bentoml configuration for Lambda with EFS
config = {
    "lambda_config": {
        "memory_size": 3008,  # 3 GB
        "timeout": 60,
        "vpc_config": {
            "subnet_ids": ["subnet-abc123"],
            "security_group_ids": ["sg-abc123"],
        },
        "file_system_config": {
            "arn": "arn:aws:elasticfilesystem:us-east-1:123:fs/fs-abc",
            "local_mount_path": "/mnt/models",
        },
    },
}

# Models loaded from EFS instead of image
bentoml.sklearn.save_model("iris_classifier", model, signatures=...)
```

---

## 3. AWS SageMaker — Managed Inference

```bash
# Deploy to SageMaker
bentoml deploy iris_service:latest --platform sagemaker \
    --region us-east-1 \
    --instance-type ml.m5.large \
    --initial-instance-count 1 \
    --max-instance-count 5
```

This creates:
- SageMaker Endpoint with auto-scaling
- Multi-AZ deployment
- CloudWatch metrics

### 3.1 SageMaker Patterns

| Pattern | Use case |
|---------|----------|
| **Real-time endpoint** | Online inference with auto-scaling |
| **Batch transform** | Offline batch inference |
| **Multi-model endpoint** | Multiple models on one endpoint |
| **Serverless inference** | Pay-per-request; cold-start |

### 3.2 Multi-Model Endpoint

```python
# BentoML serves multiple models on one endpoint
# SageMaker routes based on target_model parameter

response = sagemaker_runtime.invoke_endpoint(
    EndpointName="my-multimodel-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "inputs": [...],
        "target_model": "iris_classifier",  # route to specific model
    }),
)
```

The same endpoint serves multiple models; BentoML handles the routing.

---

## 4. Google Cloud Run

```bash
# Build and deploy
bentoml containerize iris_service:latest
gcloud run deploy iris-service \
    --image iris_service:latest \
    --region us-central1 \
    --platform managed \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --allow-unauthenticated
```

Cloud Run features:
- **Pay-per-request** — scales to zero when idle
- **Auto-scaling** — 0 to N instances based on traffic
- **HTTPS** — automatic
- **Custom domains** — map your own domain

---

## 5. Azure ML

```bash
# Deploy to Azure ML
bentoml deploy iris_service:latest --platform azure-ml \
    --workspace-name my-workspace \
    --aci-name iris-service \
    --instance-type Standard_D2_v3
```

Azure ML features:
- **Managed endpoints** — managed inference infrastructure
- **Online + batch** — both deployment modes
- **MLOps integration** — Azure DevOps, GitHub Actions
- **Hybrid** — on-prem + cloud

---

## 6. Docker Compose — Local Dev

```yaml
# docker-compose.yml
services:
  iris-service:
    image: iris_service:latest
    ports:
      - "3000:3000"
    environment:
      - LOG_LEVEL=info
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/healthz"]
      interval: 30s
      timeout: 5s
      retries: 3
```

```bash
docker compose up -d
```

For local dev only; not for production.

---

## 7. Choosing the Right Target

| Workload | Best target | Why |
|----------|-------------|-----|
| **Bursty unpredictable traffic** | Lambda | Pay-per-request; scales to zero |
| **Steady high traffic** | Kubernetes | Cost-effective at scale |
| **AWS-native, simple** | SageMaker | Managed; integrated with AWS |
| **GCP-native, simple** | Cloud Run | Serverless; pay-per-request |
| **Enterprise hybrid** | Azure ML | On-prem + cloud |
| **Local dev / testing** | Docker Compose | Simple, no cloud |

---

## 8. Multi-Region Deployment

For global low latency, deploy to multiple regions:

```python
# Same Bento deployed to multiple regions
regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"]

for region in regions:
    bentoml deploy iris_service:latest \
        --platform aws-lambda \
        --region $region
```

A global load balancer (Cloudflare, AWS Global Accelerator) routes to the nearest region.

---

## 9. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Model

on:
  push:
    paths:
      - "models/**"

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install BentoML
        run: pip install bentoml
      
      - name: Train and save model
        run: python train.py  # writes to BentoML store
      
      - name: Build Bento
        run: bentoml build
      
      - name: Deploy to staging
        run: |
          bentoml deploy iris_service:latest \
              --platform kubernetes \
              --namespace staging
      
      - name: Run smoke tests
        run: python tests/smoke.py
      
      - name: Deploy to production
        if: github.ref == 'refs/heads/main'
        run: |
          bentoml deploy iris_service:latest \
              --platform kubernetes \
              --namespace production
```

The pipeline: train → save → build → deploy to staging → smoke test → deploy to prod.

---

## 10. Real-World Example — StayBot Recommendation

For your **StayBot** portfolio project:

```bash
# Build the StayBot recommendation Bento
bentoml build

# Deploy to Cloud Run (serverless, scales to zero when idle)
bentoml containerize staybot_rec:latest
gcloud run deploy staybot-rec \
    --image staybot_rec:latest \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 100

# Cost: $0 when idle, $0.00002400/vCPU-second when active
```

For production with auto-scaling on load: Cloud Run is ideal.

---

## 11. Antipatterns

### 11.1 Antipattern 1: Lambda for large models

```python
# ❌ 10 GB model in Lambda image; 30s cold start; 6 MB payload limit
bentoml deploy model_x:latest --platform aws-lambda

# ✅ Use Lambda for small models, SageMaker/K8s for large
```

### 11.2 Antipattern 2: No health checks

```python
# ❌ No health endpoint; K8s liveness probe fails
@bentoml.service
class MyService:
    @bentoml.api
    def predict(self, input_data):
        return self.model.predict(input_data)
    # No /healthz endpoint

# ✅ Add health check
@bentoml.service
class MyService:
    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy"}
```

### 11.3 Antipattern 3: No graceful shutdown

```python
# ❌ K8s kills pod mid-request; user sees error
# ✅ Configure graceful shutdown
# bentoml.yaml
service:
  shutdown_timeout: 30  # wait 30s for in-flight requests
```

### 11.4 Antipattern 4: Manual scaling

```python
# ❌ Set replicas manually based on load estimates
bentoml deploy --replicas 10  # guess

# ✅ Auto-scaling on metrics
# HPA configured to scale on CPU + request rate
```

### 11.5 Antipattern 5: Deploying to production without staging

```python
# ❌ Deploy to production directly
bentoml deploy --platform kubernetes --namespace production

# ✅ Staging first; smoke tests; then production
bentoml deploy --namespace staging  # smoke tests
bentoml deploy --namespace production  # after tests pass
```

---

## 🎯 Key Takeaways

- BentoML deploys to Kubernetes, Lambda, SageMaker, Cloud Run, Azure ML.
- Choose target based on traffic pattern: Lambda (bursty), K8s (steady), SageMaker (AWS-native).
- EFS for large models in Lambda; container images for small models.
- Multi-model endpoints for hosting multiple models in one endpoint.
- Multi-region deployment for global low latency.
- CI/CD pipeline: train → save → build → deploy to staging → smoke test → deploy to prod.
- Always add health checks; configure graceful shutdown.
- Avoid Lambda for large models, no health checks, no graceful shutdown, manual scaling, direct-to-prod.

## References

- BentoML Kubernetes — [docs.bentoml.com/en/latest/scale-with-kubernetes/deployment-with-kubernetes](https://docs.bentoml.com/en/latest/scale-with-kubernetes/deployment-with-kubernetes.html)
- BentoML AWS Lambda — [docs.bentoml.com/en/latest/scale-with-aws-lambda](https://docs.bentoml.com/en/latest/scale-with-aws-lambda.html)
- BentoML SageMaker — [docs.bentoml.com/en/latest/scale-with-aws-sagemaker](https://docs.bentoml.com/en/latest/scale-with-aws-sagemaker.html)
- BentoML Cloud Run — [docs.bentoml.com/en/latest/scale-with-gcp](https://docs.bentoml.com/en/latest/scale-with-gcp.html)
- [[09 - MLOps y Produccion/32 - KServe and Knative|KServe]] — K8s-native alternative
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — multi-cloud deploy
- [[09 - MLOps y Produccion/42 - BentoML Production Model Serving/05 - Capstone - Production BentoML Platform|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/22 - End-to-End ML Project|E2E ML Project]] — CI/CD patterns