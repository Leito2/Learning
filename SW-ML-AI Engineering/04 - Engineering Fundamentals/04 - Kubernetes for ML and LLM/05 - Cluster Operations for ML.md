# Cluster Operations for ML

## 1. Node Pools for ML Workloads

Separate GPU nodes from general-purpose nodes:

```bash
# Create GPU node pool
gcloud container node-pools create gpu-pool \
  --machine-type a2-highgpu-1g \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --node-taints nvidia.com/gpu=true:NoSchedule \
  --node-labels workload=gpu

# Create CPU-only node pool
gcloud container node-pools create cpu-pool \
  --machine-type n2-standard-8 \
  --node-labels workload=cpu
```

## 2. Taints and Tolerations

```bash
# Taint GPU nodes so only GPU pods schedule there
kubectl taint nodes gpu-node-1 nvidia.com/gpu=true:NoSchedule
```

Pod toleration:

```yaml
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
  nodeSelector:
    workload: gpu
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 1
```

## 3. Node Affinity for Specialized Hardware

```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu
            operator: In
            values:
            - "a100"
            - "h100"
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: workload
            operator: In
            values:
            - gpu
```

## 4. Pod Priority and Preemption

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-inference
value: 1000000
preemptionPolicy: PreemptLowerPriority

# inference pods
priorityClassName: high-priority-inference
```

Training jobs get lower priority than inference:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority-training
value: 100
```

## 5. Resource Quotas per Namespace

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: inference
spec:
  hard:
    requests.nvidia.com/gpu: 8
    limits.nvidia.com/gpu: 8
    requests.memory: "64Gi"
    limits.memory: "128Gi"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: inference
spec:
  limits:
  - max:
      nvidia.com/gpu: 2
    min:
      nvidia.com/gpu: 1
    type: Container
```

## 6. Cluster Autoscaler for GPU

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler
  namespace: kube-system
data:
  config: |
    nodeGroups:
    - name: gpu-pool
      minSize: 1
      maxSize: 10
      machineType: a2-highgpu-1g
      gpuType: nvidia-tesla-a100
      gpuCount: 1
    - name: cpu-pool
      minSize: 3
      maxSize: 20
      machineType: n2-standard-8
```

## 7. Node Maintenance for GPU Nodes

```bash
# Drain a GPU node for maintenance
kubectl drain gpu-node-1 --ignore-daemonsets --delete-emptydir-data

# After maintenance
kubectl uncordon gpu-node-1
```

For planned maintenance with PodDisruptionBudget:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: model-server
```

## 8. Cost Optimization

```bash
# Use spot/preemptible GPU instances for training
gcloud container node-pools create spot-gpu \
  --spot \
  --machine-type a2-highgpu-1g

# Add toleration for spot instance eviction
tolerations:
- key: cloud.google.com/gke-spot
  operator: Exists
  effect: NoSchedule
```

## Key Takeaways

- Node pools: separate GPU (expensive) from CPU (cheap) nodes
- Taints + tolerations: ensure GPU pods land on GPU nodes
- Node affinity: pin pods to specific GPU types (A100, H100)
- Priority classes: inference > training for production
- Resource quotas: prevent namespace from consuming all GPUs
- Cluster autoscaler: scale GPU node pool up/down based on demand
- PDB: ensure minimum replicas during node maintenance
- Spot instances: use for training (evictable), not inference (latency-sensitive)
