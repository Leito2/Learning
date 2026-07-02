# K8s Fundamentals for ML Engineers

## 1. Core Concepts Map

```
Pod (GPU container) → Deployment (replicas) → Service (stable IP) → Ingress (HTTP route) → Autoscaler
       |
   ConfigMap (env vars)
   Secret (API keys)
   PersistentVolumeClaim (model weights)
```

## 2. Pod with GPU

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference-pod
spec:
  containers:
  - name: model
    image: my-registry/inference-server:latest
    resources:
      limits:
        nvidia.com/gpu: 1
    env:
    - name: MODEL_PATH
      value: "/models/model.pt"
    volumeMounts:
    - name: model-storage
      mountPath: /models
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: models-pvc
```

## 3. Deployment + Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: server
        image: my-registry/inference:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  selector:
    app: model-server
  ports:
  - port: 80
    targetPort: 8000
```

## 4. Ingress with TLS

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: inference-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.ml-platform.com
    secretName: ml-tls
  rules:
  - host: api.ml-platform.com
    http:
      paths:
      - path: /predict
        pathType: Prefix
        backend:
          service:
            name: model-server
            port:
              number: 80
```

## 5. ConfigMap and Secrets

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_NAME: "bert-base"
  MAX_BATCH_SIZE: "32"
---
apiVersion: v1
kind: Secret
metadata:
  name: inference-secrets
type: Opaque
stringData:
  API_KEY: "sk-..."
  DB_PASSWORD: "secret123"
```

In deployment:

```yaml
envFrom:
- configMapRef:
    name: model-config
- secretRef:
    name: inference-secrets
```

## 6. Persistent Volumes for Model Weights

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-client
```

## 7. Useful Commands

```bash
kubectl get pods -l app=model-server
kubectl logs -f deployment/model-server
kubectl exec -it pod/model-server-xxx -- bash
kubectl port-forward service/model-server 8000:80
kubectl describe pod model-server-xxx  # events + errors
kubectl top pod model-server-xxx       # CPU/memory usage
```

## Key Takeaways

- GPU pod: `resources.limits.nvidia.com/gpu: 1`
- Deployment: declarative replicas with rolling updates
- Service: stable internal DNS + load balancing
- Ingress: TLS termination + path routing
- ConfigMap/Secret: env injection, avoid hardcoding
- PVC: persistent model storage across pod restarts
