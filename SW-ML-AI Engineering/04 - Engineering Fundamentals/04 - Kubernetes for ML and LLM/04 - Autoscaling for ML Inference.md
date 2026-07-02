# Autoscaling for ML Inference

## 1. Three Autoscalers

| Autoscaler | Trigger | Use Case |
|------------|---------|----------|
| **HPA** (Horizontal Pod Autoscaler) | CPU/memory/custom metrics | Stateless inference with predictable load |
| **VPA** (Vertical Pod Autoscaler) | CPU/memory recommendations | Right-sizing GPU requests |
| **KEDA** (Kubernetes Event-Driven Autoscaling) | Queue length, event count, custom | Real-time inference, batch jobs, event-driven |

## 2. HPA for Inference

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_latency_p99
      target:
        type: AverageValue
        averageValue: 500m  # 500ms p99 latency target
```

## 3. HPA with GPU Metrics (Prometheus + Custom)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpu-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: DCGM_FI_DEV_GPU_UTIL
      target:
        type: AverageValue
        averageValue: 70  # scale when GPU utilization > 70%
```

Install Prometheus Adapter:

```bash
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --set prometheus.url=http://prometheus.monitoring.svc
```

## 4. KEDA for Queue-Driven Inference

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: inference-scaler
spec:
  scaleTargetRef:
    name: model-server
  minReplicaCount: 1
  maxReplicaCount: 20
  triggers:
  - type: kafka
    metadata:
      topic: inference-requests
      bootstrapServers: kafka-cluster:9092
      lagThreshold: "50"
      consumerGroup: inference-group
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: inference_queue_depth
      query: |
        sum(kafka_consumer_lag{topic="inference-requests"})
      threshold: "100"
```

## 5. Scale-to-Zero with Knative

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: inference-server
spec:
  template:
    spec:
      containers:
      - image: my-registry/inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1
      autoscaling:
        minScale: 0  # scale-to-zero when idle
        maxScale: 10
        target: 1    # target 1 concurrent request per replica
```

Scale-to-zero saves GPU cost during idle periods. Cold start: ~10-30s for model load.

## 6. VPA for GPU Sizing

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: inference-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: server
      minAllowed:
        memory: 4Gi
      maxAllowed:
        memory: 32Gi
        nvidia.com/gpu: 1
```

VPA recommends optimal memory/CPU requests based on historical usage. Use for right-sizing GPU node pools.

## 7. Autoscaling Strategy for LLMs

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-scaler
spec:
  scaleTargetRef:
    name: vllm-server
  minReplicaCount: 2    # keep 2 warm for base load
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      metricName: vllm:running_requests
      query: |
        sum(avg_over_time(vllm:running_requests{namespace="llm"}[1m]))
      threshold: "50"
  - type: prometheus
    metadata:
      metricName: vllm:gpu_memory_used
      query: |
        avg(vllm:gpu_memory_used) / avg(vllm:gpu_memory_total) * 100
      threshold: "80"
```

## Key Takeaways

- HPA: scale on CPU/memory/latency for stateless inference
- HPA + Prometheus: scale on GPU utilization with custom metrics
- KEDA: scale on queue depth (Kafka, RabbitMQ, SQS)
- Knative: scale-to-zero for cost savings on idle services
- VPA: right-size GPU/memory requests automatically
- LLM strategy: keep warm replicas + scale on request queue + GPU memory
- Cold start mitigation: `minReplicaCount >= 1` for production workloads
