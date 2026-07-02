# GPU Scheduling

## 1. NVIDIA GPU Operator

The NVIDIA GPU Operator replaces manual driver installation with K8s-native GPU management:

```bash
helm install gpu-operator nvidia/gpu-operator --set driver.enabled=true
```

It deploys:
- **nvidia-driver-daemonset** — GPU driver on each node
- **nvidia-device-plugin** — advertises GPU count to Kubelet
- **nvidia-dcgm-exporter** — GPU metrics (memory, utilization, temperature)
- **nvidia-mig-manager** — MIG partitioning

Verify:

```bash
kubectl describe node <node> | grep nvidia.com/gpu
kubectl get pods -n gpu-operator
```

## 2. GPU Request Patterns

```yaml
spec:
  containers:
  - name: inference
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
```

Both `requests` and `limits` must equal for GPUs (no overcommit). A pod requesting `nvidia.com/gpu: 2` gets exclusive access to two GPUs.

## 3. GPU Sharing (Time-Slicing)

Share one GPU across multiple pods:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  config: |
    version: v1
    flags:
      migStrategy: "none"
    sharing:
      timeSlicing:
        replicas: 4
        resources:
        - name: nvidia.com/gpu
```

Time-slicing gives each pod 1/4 of GPU time. Good for small models, bad for latency-sensitive workloads.

## 4. MIG (Multi-Instance GPU)

Partition a single A100/H100 into isolated GPU instances:

```bash
# On the node, create MIG profiles
nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb
```

K8s resource:

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 1
        nvidia.com/mig-profile: "1g.10gb"
```

MIG profiles:
| Profile | GPU Memory | Compute Units | Use Case |
|---------|-----------|---------------|----------|
| `1g.10gb` | 10 GB | 1/7 | BERT, small transformer |
| `2g.20gb` | 20 GB | 2/7 | Medium transformer |
| `3g.40gb` | 40 GB | 3/7 | Large model |
| `4g.40gb` | 40 GB | 4/7 | Very large model |
| `7g.80gb` | 80 GB | 7/7 | Full GPU |

## 5. MIG with GPU Operator

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-config
  namespace: gpu-operator
data:
  config: |
    version: v1
    migStrategy: "mixed"
    migConfig:
      0:
        - device: 0
          migEnabled: true
          migDevices:
            1g.10gb: 4
```

## 6. GPU Monitoring

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
spec:
  endpoints:
  - port: metrics
    path: /metrics
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
```

Prometheus metrics from DCGM:
- `DCGM_FI_DEV_GPU_UTIL` — GPU utilization %
- `DCGM_FI_DEV_FB_USED` — memory used bytes
- `DCGM_FI_DEV_POWER_USAGE` — power draw watts
- `DCGM_FI_DEV_MEM_COPY_UTIL` — memory copy utilization

## 7. GPU Error Recovery

```yaml
spec:
  containers:
  - name: inference
    resources:
      limits:
        nvidia.com/gpu: 1
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "all"
---
# Pod tolerates GPU failure with restartPolicy
restartPolicy: OnFailure
```

## Key Takeaways

- NVIDIA GPU Operator: drivers, device plugin, MIG, monitoring in one Helm chart
- Time-slicing: share GPU across pods (4:1 ratio), good for small models
- MIG: hardware partitioning (A100/H100), full isolation but static
- DCGM exporter: GPU metrics for Prometheus (utilization, memory, power)
- GPU requests = limits (no overcommit)
- `kubectl describe node` shows allocatable GPUs per node
