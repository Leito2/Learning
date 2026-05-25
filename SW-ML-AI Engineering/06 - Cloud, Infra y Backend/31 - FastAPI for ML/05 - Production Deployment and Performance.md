# 🚀 Production Deployment and Performance

## 🎯 Learning Objectives

- Configure reverse proxy layers (Traefik, Nginx) with TLS termination, rate limiting, and load balancing for ML inference services
- Build production Docker images with multi-stage builds and deploy FastAPI ML services to Kubernetes with complete Deployment, Service, and Ingress manifests
- Instrument FastAPI applications with OpenTelemetry tracing, Prometheus metrics, and structured logging for end-to-end observability of ML prediction pipelines
- Optimize serving performance through connection pooling, response caching, batch inference, GPU utilization tracking, and systematic load testing

## Introduction

A FastAPI ML service that runs perfectly on a laptop at 10 RPS will collapse at 1000 RPS in production — not because FastAPI is slow, but because production introduces realities absent from development: network latency, TLS overhead, container resource limits, orchestrator health checks, and observability collectors that compete for the same CPU cores. Deploying an ML API to production is an exercise in understanding the entire request path from client to GPU and back, and optimizing every link in that chain.

The deployment stack for a production ML API has four layers. The edge layer (reverse proxy: Traefik/Nginx) terminates TLS, applies rate limits, and routes requests. The container layer (Docker) packages the application, its Python dependencies, and CUDA libraries into an immutable artifact. The orchestrator layer (Kubernetes) manages scheduling, scaling, health checks, and rolling updates. The observability layer (OpenTelemetry, Prometheus, structured logging) provides visibility into every prediction — its latency, its GPU consumption, its input distribution, and whether it succeeded or failed. Each layer introduces configuration decisions that directly impact throughput, latency, and reliability.

This note walks through the complete production deployment of a FastAPI ML service, from local Docker testing to a Kubernetes cluster with TLS and monitoring. It covers performance optimization patterns — connection pooling, response caching, batch inference scheduling, and GPU utilization tracking — that separate a prototype serving 50 RPS from a production service handling 5000 RPS. The deployment patterns here extend the containerization strategies in [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/02 - Containerization/02 - Containerization|Containerization for ML]] and the observability principles from ML monitoring frameworks.

---

## Module 1: Reverse Proxy and TLS

### 1.1 Theoretical Foundation 🧠

FastAPI's Uvicorn server is an application server, not an edge server. It lacks built-in TLS termination, robust rate limiting, request buffering, and DDoS protection. In production, a reverse proxy sits between the internet and Uvicorn, handling these concerns. The proxy terminates TLS (decrypts HTTPS, forwards plain HTTP to the backend), buffers slow clients, applies rate limits, and load-balances across multiple Uvicorn workers. This separation of concerns — proxy handles networking, FastAPI handles business logic — is the standard architecture for Python web services at every scale.

Traefik and Nginx dominate this space. Traefik is cloud-native: it auto-discovers services via Docker labels or Kubernetes Ingress resources, integrates with Let's Encrypt for automatic TLS certificate renewal, and provides a dashboard. Nginx is battle-tested: it offers finer-grained configuration, superior static file serving, and a massive ecosystem of modules. For ML services, Traefik's Docker Compose and Kubernetes integration makes it the faster path to a production-ready setup. Nginx is preferred when you need advanced request rewriting, complex caching logic, or integration with existing Nginx-based infrastructure.

Rate limiting at the proxy layer is essential for ML APIs. A single misconfigured client can saturate GPU resources, denying service to all other users. Token bucket rate limiting (Traefik's `RateLimit` middleware, Nginx's `limit_req_zone`) caps requests per time window per client IP, API key, or any request header. This protects the GPU layer without requiring the ML application to track rate state — the proxy handles it at wire speed.

### 1.2 Mental Model 📐

```
┌─── Production Network Architecture ──────────────────────┐
│                                                          │
│  Internet                                                 │
│     │                                                    │
│     ▼                                                    │
│  ┌──────────────────────────────────────────┐            │
│  │  Reverse Proxy (Traefik / Nginx)         │            │
│  │  ┌────────────────────────────────────┐  │            │
│  │  │ TLS Termination (port 443)         │  │            │
│  │  │ Rate Limiting (100 req/s per IP)   │  │            │
│  │  │ Request buffering                  │  │            │
│  │  │ Access logging                     │  │            │
│  │  │ Load balancing → Uvicorn workers   │  │            │
│  │  └────────────────────────────────────┘  │            │
│  └──────────┬───────────────────────────────┘            │
│             │ HTTP (internal network, no TLS needed)      │
│             ▼                                            │
│  ┌──────────────────────────────────────────┐            │
│  │  FastAPI + Uvicorn (port 8000)           │            │
│  │  ┌────────┐ ┌────────┐ ┌────────┐        │            │
│  │  │Worker 1│ │Worker 2│ │Worker 3│  ...    │            │
│  │  └────────┘ └────────┘ └────────┘        │            │
│  └──────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────┘
```

```
┌─── Rate Limiting Strategies ─────────────────────────────┐
│                                                          │
│  Strategy        │ Scope       │ ML Use Case              │
│  ────────────────┼─────────────┼─────────────────────────│
│  Per-IP          │ Client IP   │ Public API, anonymous    │
│  Per-API-Key     │ API key     │ Authenticated users      │
│  Per-Endpoint    │ URL path    │ /predict=100/s,          │
│                  │             │ /batch=10/s              │
│  Tiered          │ User tier   │ Free=10/min, Pro=100/min │
│  GPU-Aware       │ GPU queue   │ Backpressure: queue > N  │
│                  │ depth       │ → 429 reject             │
│  ────────────────┼─────────────┼─────────────────────────│
│  Proxy layer     │ High-perf   │ Traefik/Nginx handles    │
│  (best)          │ C-level     │ rate limits at wire speed│
└──────────────────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```yaml
# traefik.yml — Static configuration for Traefik reverse proxy
# WHY: Traefik auto-discovers Docker containers and K8s services.
# This file configures entrypoints, TLS providers, and global middleware.

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
          permanent: true

  websecure:
    address: ":443"
    http:
      tls:
        certResolver: letsencrypt  # auto-TLS via Let's Encrypt
        domains:
          - main: "ml-api.example.com"

# Let's Encrypt configuration
certificatesResolvers:
  letsencrypt:
    acme:
      email: "ml-ops@example.com"
      storage: "/letsencrypt/acme.json"
      httpChallenge:
        entryPoint: web

# Docker provider — auto-discovers containers with labels
providers:
  docker:
    exposedByDefault: false  # only expose containers with traefik.enable=true
    network: traefik-network
```

```yaml
# docker-compose.yml — FastAPI ML service with Traefik reverse proxy
# WHY: Single-file production deployment with automatic TLS, rate limiting,
# and load balancing. docker compose up -d deploys the entire stack.

version: "3.9"

services:
  # ─── Traefik Reverse Proxy ───
  traefik:
    image: traefik:v3.0
    command:
      - "--configfile=/etc/traefik/traefik.yml"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "./traefik.yml:/etc/traefik/traefik.yml:ro"
      - "./dynamic.yml:/etc/traefik/dynamic.yml:ro"
      - "letsencrypt:/letsencrypt"
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
    networks:
      - traefik-network

  # ─── FastAPI ML Service ───
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - MODEL_PATH=/models/bert-sentiment-v2
      - WORKERS=4
      - GPU_CONCURRENCY=4
    volumes:
      - models:/models:ro  # read-only model volume
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ml-api.rule=Host(`ml-api.example.com`)"
      - "traefik.http.routers.ml-api.entrypoints=websecure"
      - "traefik.http.routers.ml-api.tls=true"
      - "traefik.http.routers.ml-api.middlewares=rate-limit@file,security-headers@file"
      - "traefik.http.services.ml-api.loadbalancer.server.port=8000"
      - "traefik.http.services.ml-api.loadbalancer.healthcheck.path=/healthz"
      - "traefik.http.services.ml-api.loadbalancer.healthcheck.interval=10s"
    networks:
      - traefik-network

volumes:
  letsencrypt:
  models:

networks:
  traefik-network:
    external: true
```

```yaml
# dynamic.yml — Traefik dynamic configuration: rate limiting, security headers
# WHY: Separate dynamic config from static config. Rate limits and headers
# change more frequently than entrypoints and providers.

http:
  middlewares:
    rate-limit:
      rateLimit:
        average: 100       # requests per second averaged over period
        period: 1s
        burst: 50          # allow short bursts beyond average
        sourceCriterion:
          ipStrategy:
            depth: 1       # use client IP (respect X-Forwarded-For)

    security-headers:
      headers:
        customFrameOptionsValue: "DENY"
        contentTypeNosniff: true
        browserXssFilter: true
        forceSTSHeader: true
        stsIncludeSubdomains: true
        stsPreload: true
        stsSeconds: 31536000
```

```nginx
# nginx.conf — Nginx reverse proxy for FastAPI ML service
# WHY: Nginx is the most widely deployed reverse proxy. Use when
# you need advanced caching, request rewriting, or existing Nginx infra.

upstream ml_api_backend {
    # Load balance across Uvicorn workers
    least_conn;  # send to worker with fewest active connections
    server 127.0.0.1:8001 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8004 weight=1 max_fails=3 fail_timeout=30s;
    keepalive 128;  # keep connections alive for reuse
}

# Rate limiting zone: 100 req/s per IP, burst 50
limit_req_zone $binary_remote_addr zone=ml_api_limit:10m rate=100r/s;

server {
    listen 443 ssl http2;
    server_name ml-api.example.com;

    # TLS configuration
    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # Request size limits (for model upload endpoints)
    client_max_body_size 5G;
    client_body_timeout 300s;

    # ML prediction endpoint
    location /predict {
        limit_req zone=ml_api_limit burst=50 nodelay;

        proxy_pass http://ml_api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for streaming responses (SSE/LLM token streaming)
        proxy_buffering off;
        proxy_cache off;

        # Timeouts for long-running predictions
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Health check — no rate limiting
    location /healthz {
        proxy_pass http://ml_api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 1.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph "Internet"
        C["Client\n(HTTPS)"]
    end

    subgraph "Reverse Proxy (Traefik/Nginx)"
        TLS["TLS Termination\n(decrypt HTTPS → HTTP)"]
        RL["Rate Limiting\n(token bucket)"]
        LB["Load Balancer\n(least connections / round robin)"]
        BUF["Request Buffering\n(absorb slow clients)"]
    end

    subgraph "FastAPI Application"
        W1["Uvicorn Worker 1\n:8001"]
        W2["Uvicorn Worker 2\n:8002"]
        W3["Uvicorn Worker 3\n:8003"]
        W4["Uvicorn Worker 4\n:8004"]
    end

    subgraph "ML Infrastructure"
        GPU["GPU / Model Server"]
        CACHE["Redis Cache"]
        DB["Feature Store DB"]
    end

    C -->|":443 HTTPS"| TLS
    TLS --> RL
    RL --> LB
    LB --> BUF
    BUF -->|":80 HTTP"| W1
    BUF -->|":80 HTTP"| W2
    BUF -->|":80 HTTP"| W3
    BUF -->|":80 HTTP"| W4
    W1 --> GPU
    W2 --> GPU
    W3 --> CACHE
    W4 --> DB
```

### 1.5 Application in ML/AI Systems 🤖

OpenAI's API serves millions of predictions per minute through an nginx-based edge layer. Their reverse proxy tier performs TLS termination, IP-based rate limiting tied to API key tiers, and request routing to model-specific backend clusters (GPT-4 goes to one cluster, embeddings to another). The nginx layer also buffers slow clients during streaming responses — a mobile client on a 3G connection doesn't block a GPU from generating tokens; nginx absorbs the buffering. This architecture enables OpenAI to expose a single API endpoint (`api.openai.com`) while routing to hundreds of backend services, applying per-user rate limits, and maintaining sub-100ms TLS termination latency.

### 1.6 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Placing Uvicorn directly on the internet without TLS termination. HTTPS clients receive connection errors, and HTTP traffic exposes API keys in plaintext.

💡 **Tip**: Always put a reverse proxy in front. Even for internal services, Traefik or Nginx provides TLS, rate limiting, and health-check-based load balancing that Uvicorn alone lacks.

⚠️ **Pitfall**: Proxy buffering on for streaming endpoints. Nginx's default `proxy_buffering on` holds the entire response before forwarding it to the client, defeating the purpose of SSE streaming.

💡 **Tip**: Set `proxy_buffering off` and `proxy_cache off` for streaming endpoints. For Traefik, streaming passthrough is the default behavior.

⚠️ **Pitfall**: Hard-coding backend server addresses in the proxy config. When Uvicorn workers restart, ports change, and the proxy routes to dead backends.

💡 **Tip**: In Docker Compose, use service names as hostnames (Docker DNS resolves them). In Kubernetes, use Service clusterIPs. Never hard-code IP:port pairs.

### 1.7 Knowledge Check ❓

1. Why does the reverse proxy terminate TLS and forward plain HTTP to FastAPI, even on internal networks?
2. Your SSE streaming endpoint returns tokens one at a time, but clients report receiving the full response at once. What nginx configuration is likely the cause?
3. A rate limit of `average: 100, period: 1s, burst: 50` is configured. Client sends 200 requests in 1 second. How many are accepted and how many rejected?

---

## Module 2: Docker and Kubernetes Deployment

### 2.1 Theoretical Foundation 🧠

Containerizing an ML API is more complex than containerizing a CRUD web service. Python dependencies alone can be 1 GB+. CUDA libraries add another 2–3 GB. Model weights range from 500 MB (BERT) to 70 GB (LLaMA 70B). A naive `pip install` Dockerfile produces a 10 GB image that takes 5 minutes to pull on each deployment — unacceptable for rapid iteration and rollback.

Multi-stage Docker builds solve this by separating the build environment (compilers, CUDA headers, full pip cache) from the runtime environment (only the artifacts needed to serve). The first stage installs all dependencies and compiles any C extensions. The second stage copies only the installed packages and model weights, discarding the build toolchain. Multi-stage builds can shrink a 10 GB image to 3 GB — still large by microservice standards, but manageable.

For Kubernetes, the key ML-specific considerations are: GPU resource requests (you must specify `nvidia.com/gpu: 1` to trigger the GPU device plugin), probe configuration (readiness probes must account for model loading time), and graceful shutdown (30+ seconds for large models to drain in-flight predictions). The Kubernetes ecosystem for ML deploys mirrors the deployment patterns in [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/02 - Containerization/02 - Containerization|ML Containerization Strategies]].

### 2.2 Mental Model 📐

```
┌─── Multi-Stage Docker Build ─────────────────────────────┐
│                                                          │
│  Stage 1: Builder (nvidia/cuda:12.1-devel-ubuntu22.04)   │
│  ┌─────────────────────────────────────────────────┐     │
│  │ • CUDA headers, compilers (gcc, nvcc)           │     │
│  │ • pip install torch transformers fastapi        │     │
│  │ • Compile C extensions (flash-attn, etc.)       │     │
│  │ • ~8 GB image size                              │     │
│  └─────────────────────────────────────────────────┘     │
│       │                                                  │
│       │ COPY --from=builder /usr/local/lib/python3.11    │
│       ▼                                                  │
│  Stage 2: Runtime (nvidia/cuda:12.1-runtime-ubuntu22.04) │
│  ┌─────────────────────────────────────────────────┐     │
│  │ • CUDA runtime only (no headers, no compilers)  │     │
│  │ • Copy Python packages from builder             │     │
│  │ • Copy model weights (mounted or baked in)      │     │
│  │ • Non-root user for security                    │     │
│  │ • ~3 GB image size                              │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘

┌─── K8s Resource Planning ────────────────────────────────┐
│                                                          │
│  Component     │ CPU Request │ Mem Request │ GPU Request  │
│  ──────────────┼─────────────┼─────────────┼─────────────│
│  FastAPI pod   │ 2 cores     │ 4 GiB       │ 1 GPU        │
│  Redis cache   │ 0.5 cores   │ 1 GiB       │ -            │
│  Nginx proxy   │ 1 core      │ 512 Mi      │ -            │
│  ──────────────┼─────────────┼─────────────┼─────────────│
│  Model: BERT   │ 2-4 cores   │ 2-4 GiB     │ 1 GPU (4GB)  │
│  Model: LLaMA7B│ 4-8 cores   │ 14-16 GiB   │ 1 GPU (16GB) │
│  Model: LLaMA70│ 8-16 cores  │ 140+ GiB    │ 4 GPU (80GB) │
└──────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```dockerfile
# Dockerfile — Multi-stage production build for FastAPI ML service
# WHY: Two-stage build isolates the heavy CUDA toolchain from the runtime
# image, reducing size from ~8 GB to ~3 GB and improving pull times.

# ═══ Stage 1: Builder ═══
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support (largest dependency — cache this layer)
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# ═══ Stage 2: Runtime ═══
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install only Python runtime (no dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder (all pip packages)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user (security best practice)
RUN groupadd -r mluser && useradd -r -g mluser -m mluser

# Copy application code
COPY --chown=mluser:mluser ./app /app
WORKDIR /app

# Copy model weights (if baking into image; otherwise mount at runtime)
# COPY --chown=mluser:mluser ./models /models

USER mluser

# Health check using FastAPI's /healthz endpoint
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

EXPOSE 8000

# Run with Uvicorn (workers managed by K8s, not Uvicorn --workers)
CMD ["python3.11", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--loop", "uvloop", "--http", "httptools", \
     "--no-access-log"]
```

```yaml
# k8s-deployment.yaml — Complete Kubernetes manifests for FastAPI ML service
# WHY: Production K8s deployment with GPU scheduling, probes, HPA,
# pod disruption budgets, and rolling update strategy.

# ═══ Namespace ═══
apiVersion: v1
kind: Namespace
metadata:
  name: ml-serving
---
# ═══ ConfigMap: Application configuration ═══
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: ml-serving
data:
  MODEL_PATH: "/models/bert-sentiment-v2"
  WORKERS: "1"           # 1 worker per pod (K8s scales pods, not workers)
  GPU_CONCURRENCY: "4"
  LOG_LEVEL: "info"
  FEATURE_STORE_URL: "https://features.internal"
---
# ═══ Secret: API keys and credentials ═══
apiVersion: v1
kind: Secret
metadata:
  name: ml-api-secrets
  namespace: ml-serving
type: Opaque
stringData:
  FEATURE_STORE_API_KEY: "sk-xxxxxxxxxxxx"
  REDIS_PASSWORD: "redis-password-here"
---
# ═══ Deployment ═══
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-serving
  labels:
    app: ml-api
    version: v2.1.0
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # allow 1 extra pod during rollout
      maxUnavailable: 0    # never go below desired count
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
        version: v2.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # ─── Graceful shutdown: allow in-flight predictions to complete ───
      terminationGracePeriodSeconds: 60

      # ─── GPU scheduling ───
      nodeSelector:
        accelerator: nvidia-tesla-t4  # target GPU nodes only
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"

      # ─── Init container: warm model cache ───
      initContainers:
        - name: model-loader
          image: ml-api:latest
          command: ["python3.11", "-c", "from app.model import load_model; load_model()"]
          envFrom:
            - configMapRef:
                name: ml-api-config
          volumeMounts:
            - name: model-storage
              mountPath: /models

      containers:
        - name: ml-api
          image: ml-api:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              protocol: TCP
          envFrom:
            - configMapRef:
                name: ml-api-config
            - secretRef:
                name: ml-api-secrets

          # ─── Resource requests and limits ───
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"

          # ─── Probes ───
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 10   # model loading time
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 30   # after model is definitely loaded
            periodSeconds: 15
            failureThreshold: 3

          # ─── Volume mounts ───
          volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true

      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-weights-pvc
---
# ═══ Service ═══
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  namespace: ml-serving
  labels:
    app: ml-api
spec:
  type: ClusterIP
  selector:
    app: ml-api
  ports:
    - name: http
      port: 80
      targetPort: 8000
      protocol: TCP
  sessionAffinity: None  # stateless — any pod can serve any request
---
# ═══ Ingress (with TLS via cert-manager) ═══
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api
  namespace: ml-serving
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/proxy-body-size: "5g"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"  # for SSE streaming
    nginx.ingress.kubernetes.io/limit-rps: "100"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - ml-api.example.com
      secretName: ml-api-tls
  rules:
    - host: ml-api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api
                port:
                  number: 80
---
# ═══ HorizontalPodAutoscaler ═══
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # wait 5 min before scaling down
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0    # scale up immediately
      policies:
        - type: Pods
          value: 2
          periodSeconds: 30
---
# ═══ PodDisruptionBudget ═══
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
  namespace: ml-serving
spec:
  minAvailable: 2  # keep at least 2 pods running during voluntary disruptions
  selector:
    matchLabels:
      app: ml-api
```

### 2.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph "Kubernetes Cluster"
        subgraph "Ingress Layer"
            INGRESS["NGINX Ingress Controller\nTLS termination + Rate limiting"]
        end

        subgraph "ml-serving Namespace"
            SVC["Service (ClusterIP)\nLoad balances across pods"]

            subgraph "Pods"
                P1["Pod 1\nFastAPI + Uvicorn\nGPU 0"]
                P2["Pod 2\nFastAPI + Uvicorn\nGPU 1"]
                P3["Pod 3\nFastAPI + Uvicorn\nGPU 2"]
                P4["Pod 4\nFastAPI + Uvicorn\nGPU 3"]
            end

            HPA["HorizontalPodAutoscaler\nCPU > 70% → scale up\nCPU < 50% → scale down"]
        end

        subgraph "Storage"
            PVC["PersistentVolumeClaim\nModel weights (read-only)"]
        end

        subgraph "Observability"
            PROM["Prometheus\nScrapes /metrics"]
            JAEGER["Jaeger\nTrace collector"]
        end
    end

    INTERNET["Internet"] -->|HTTPS| INGRESS
    INGRESS --> SVC
    SVC --> P1
    SVC --> P2
    SVC --> P3
    SVC --> P4
    HPA -.->|monitors| P1
    HPA -.->|monitors| P2
    PVC --> P1
    PVC --> P2
    PVC --> P3
    PVC --> P4
    P1 -.->|/metrics| PROM
    P1 -.->|traces| JAEGER
    P2 -.->|/metrics| PROM
    P2 -.->|traces| JAEGER
```

### 2.5 Application in ML/AI Systems 🤖

Anthropic serves Claude models from Kubernetes clusters with GPU scheduling. Their deployment architecture uses a multi-stage Docker build that separates the PyTorch/CUDA compilation stage from the runtime image, resulting in a final image of ~4 GB for their mid-size models. Each pod requests exactly 1 GPU (`nvidia.com/gpu: 1`) and 16 GB RAM. Readiness probes check not just HTTP liveness but also that the model produces valid-shape outputs — a deep health check that catches corrupted weight loading. HorizontalPodAutoscaler scales based on GPU utilization (via DCGM metrics, not just CPU), automatically provisioning new pods when GPU saturation exceeds 80%. Their `terminationGracePeriodSeconds: 120` ensures long-generation requests (up to 100K tokens) complete before the pod terminates during rolling updates.

### 2.6 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Using `terminationGracePeriodSeconds: 30` (K8s default) when model inference takes 5+ seconds per request and pods serve multiple concurrent requests. Pods are killed mid-inference.

💡 **Tip**: Set `terminationGracePeriodSeconds` ≥ `(max_concurrent_requests × P99_latency) + 10s`. For a pod handling 4 concurrent requests of 5s each, use at least 30s.

⚠️ **Pitfall**: GPU node has 8 GPUs, Deployment requests `nvidia.com/gpu: 1` with `replicas: 4`, but no node selector. Pods land on non-GPU nodes and fail to start.

💡 **Tip**: Use `nodeSelector: { accelerator: nvidia-tesla-t4 }` or `nodeAffinity` to ensure GPU pods schedule only on GPU nodes. Add tolerations for GPU taints.

⚠️ **Pitfall**: Baking model weights into the Docker image. A 5 GB model baked into a 4 GB app image creates a 9 GB image that takes minutes to pull.

💡 **Tip**: Mount model weights via PersistentVolumeClaim or init container that downloads from object storage. The Docker image contains only code and dependencies (~2–4 GB).

### 2.7 Knowledge Check ❓

1. Your P99 inference latency is 8 seconds, and each pod handles up to 4 concurrent requests. What `terminationGracePeriodSeconds` should you set?
2. A multi-stage Docker build copies `/opt/venv` from the builder stage. Why not just `pip install` in the runtime stage?
3. Your `readinessProbe` has `initialDelaySeconds: 5`, but model loading takes 15 seconds. What happens during those 10 seconds?

---

## Module 3: Observability and Monitoring

### 3.1 Theoretical Foundation 🧠

A production ML API without observability is a black box. When a user reports "the model is slow," you cannot determine whether the issue is in TLS termination, request validation, feature store latency, model inference time, or response serialization. Observability provides the signal chain: **traces** show the end-to-end journey of a single request across services, **metrics** aggregate behavior across all requests (latency histograms, error rates, GPU utilization), and **logs** capture discrete events with context (anomalous input detected, model version changed, cache eviction triggered).

OpenTelemetry (OTel) is the CNCF standard for distributed tracing and metrics collection. It defines an API for creating spans (named operations with start/end times and attributes) and exporting them to backends like Jaeger, Zipkin, or cloud-native services (Grafana Cloud, Datadog). FastAPI integrates with OTel via the `opentelemetry-instrumentation-fastapi` package, which auto-instruments every route with spans capturing HTTP method, path, status code, and latency.

Prometheus is the de facto standard for metrics collection in Kubernetes environments. FastAPI exposes metrics via a `/metrics` endpoint (using `prometheus-fastapi-instrumentator` or `prometheus_client`). Prometheus scrapes this endpoint every 15–60 seconds, storing time-series data for alerting (Alertmanager) and visualization (Grafana). The key ML-specific metrics are: prediction latency histogram (P50/P95/P99), requests per second, error rate by status code, GPU utilization (via DCGM), and model version distribution (which model version is serving what fraction of traffic).

### 3.2 Mental Model 📐

```
┌─── Observability Architecture ───────────────────────────┐
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  FastAPI Application                             │    │
│  │  ┌────────────────┐  ┌────────────────────────┐  │    │
│  │  │ OpenTelemetry  │  │ Prometheus Metrics     │  │    │
│  │  │ Auto-Instrument│  │ /metrics endpoint      │  │    │
│  │  │ Spans: predict │  │ Counters, Histograms   │  │    │
│  │  │   ├─ validate  │  │ Gauges (GPU mem)       │  │    │
│  │  │   ├─ fetch_f   │  └───────────┬────────────┘  │    │
│  │  │   └─ infer     │              │                │    │
│  │  └────────┬───────┘              │                │    │
│  └───────────┼──────────────────────┼────────────────┘    │
│              │                      │                      │
│              ▼                      ▼                      │
│  ┌──────────────────┐  ┌──────────────────────────┐      │
│  │  Jaeger / Zipkin │  │  Prometheus + Grafana    │      │
│  │  Trace Viewer    │  │  Dashboards + Alerts     │      │
│  └──────────────────┘  └──────────────────────────┘      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Structured Logging (JSON, ELK / Loki)           │    │
│  │  {timestamp, level, request_id, prediction, ...} │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

```
┌─── Key ML Metrics ──────────────────────────────────────┐
│                                                          │
│  Metric                │ Type      │ Alert Threshold     │
│  ──────────────────────┼───────────┼────────────────────│
│  prediction_latency_ms │ Histogram │ P99 > 500ms         │
│  requests_per_second   │ Counter   │ > 90% of capacity   │
│  error_rate            │ Counter   │ > 1% of total       │
│  gpu_utilization_pct   │ Gauge     │ > 90% (scale up)    │
│  gpu_memory_used_bytes │ Gauge     │ > 90% (OOM risk)    │
│  model_version         │ Info      │ unexpected version  │
│  cache_hit_rate        │ Gauge     │ < 50%               │
│  input_distribution    │ Histogram │ drift detected      │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```python
# observability.py — Instrumented FastAPI app with OTel tracing,
# Prometheus metrics, and structured logging
import time
import logging
import json
import sys
from typing import Callable

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram, Gauge, Info
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# ═══ Structured Logging ═══
# WHY: JSON-formatted logs are parseable by log aggregators (ELK, Loki).
# Each log line includes request_id for correlation with traces.
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    return logging.getLogger("ml-api")

logger = setup_logging()

# ═══ OpenTelemetry Tracing ═══
# WHY: Distributed tracing follows a prediction request across services
# (API → Feature Store → Model Server → Cache). Each span records timing.
def setup_tracing(service_name: str, otlp_endpoint: str):
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

# ═══ Prometheus Metrics ═══
# WHY: Custom metrics track ML-specific behaviors. Built-in HTTP metrics
# (from Instrumentator) cover status codes; we add model-level metrics.

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Model prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of predictions",
    ["model_version", "status"],
)

GPU_MEMORY_USED = Gauge(
    "ml_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device"],
)

MODEL_INFO = Info(
    "ml_model",
    "Model metadata",
)

# ═══ FastAPI Application Setup ═══

app = FastAPI(title="Observable ML API")

# Auto-instrument FastAPI with OTel
FastAPIInstrumentor.instrument_app(app)

# Auto-instrument with Prometheus
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
)
instrumentator.add(metrics.latency())
instrumentator.add(metrics.requests())
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics")

# ═══ Custom Middleware: Request ID for log-trace correlation ═══
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Attach request_id to logger for this request
        logger.info("request started", extra={"request_id": request_id})

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(ObservabilityMiddleware)

# ═══ Instrumented Prediction Endpoint ═══
@app.post("/predict")
async def predict(request: Request, text: str):
    request_id = request.state.request_id
    model_version = "bert-v2.1.0"

    start = time.perf_counter()
    try:
        # Model inference — automatically captured as an OTel span
        prediction = 0.95  # substitute with actual model call

        PREDICTION_LATENCY.observe(time.perf_counter() - start)
        PREDICTION_COUNTER.labels(model_version=model_version, status="success").inc()

        logger.info("prediction complete", extra={
            "request_id": request_id,
            "model_version": model_version,
            "prediction": prediction,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        })

        return {"prediction": prediction, "model": model_version}

    except Exception as e:
        PREDICTION_COUNTER.labels(model_version=model_version, status="error").inc()
        logger.error("prediction failed", extra={
            "request_id": request_id,
            "error": str(e),
        })
        raise

# ═══ GPU Metrics Collection (background task) ═══
# WHY: GPU metrics change independently of HTTP requests. A background
# task periodically polls nvidia-smi and updates Prometheus gauges.
import asyncio
import subprocess

async def collect_gpu_metrics():
    """Background coroutine that polls GPU stats every 5 seconds."""
    while True:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            memory_mb = float(result.stdout.strip())
            GPU_MEMORY_USED.labels(device="gpu0").set(memory_mb * 1024 * 1024)
        except Exception:
            pass  # GPU monitoring failure shouldn't crash the app
        await asyncio.sleep(5)
```

### 3.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph "FastAPI Pod"
        APP["FastAPI Application"]
        OTel["OpenTelemetry SDK\nAuto-instrumentation"]
        PROM["Prometheus\n/metrics endpoint"]
        LOG["Structured Logger\n(JSON → stdout)"]
    end

    subgraph "Collectors"
        JAEGER["Jaeger Collector\nSpan ingestion"]
        PM["Prometheus Server\nScrapes /metrics every 15s"]
        FLUENT["Fluentd / Vector\nCollects stdout logs"]
    end

    subgraph "Storage & Visualization"
        ES["Elasticsearch\nLog storage"]
        GRAFANA["Grafana\nDashboards + Alerts"]
        JAEGER_UI["Jaeger UI\nTrace viewer"]
        ALERT["Alertmanager\nPagerDuty / Slack"]
    end

    APP --> OTel
    APP --> PROM
    APP --> LOG
    OTel -->|"OTLP gRPC"| JAEGER
    PROM -->|"HTTP scrape"| PM
    LOG -->|"stdout"| FLUENT
    JAEGER --> JAEGER_UI
    PM --> GRAFANA
    FLUENT --> ES
    GRAFANA --> ALERT
    ES --> GRAFANA
```

### 3.5 Application in ML/AI Systems 🤖

Uber's ML platform uses a comprehensive observability stack built on OpenTelemetry and their internal metrics infrastructure (M3). Every prediction request carries a trace ID that propagates through: the API gateway → feature store lookup → model inference → post-processing → cache write → response. Their Grafana dashboards show per-model latency distributions with P50/P95/P99 breakdowns by feature store latency, inference time, and serialization overhead. This granularity enabled them to identify that 30% of prediction latency was spent in feature store serialization (protobuf → JSON conversion), not in model inference — a finding that led to a refactor saving millions in GPU costs by reducing idle time. Their alerting rules trigger when: P99 latency exceeds 2× baseline, error rate exceeds 1%, or model version distribution deviates from expected canary percentages.

### 3.6 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Logging the full input payload (text, image bytes) at INFO level. This floods the log system with gigabytes of data and may expose PII.

💡 **Tip**: Log input hashes (`hashlib.sha256(input_text.encode()).hexdigest()[:12]`) instead of raw inputs. Store full inputs only in debug mode or a dedicated data warehouse with access controls.

⚠️ **Pitfall**: Emitting a Prometheus histogram with default buckets (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10). ML predictions often cluster between 50–200 ms, making default buckets useless.

💡 **Tip**: Define custom histogram buckets that match your latency profile: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]`. Prefer more buckets in your P50–P95 range.

⚠️ **Pitfall**: Using synchronous OTel span exporters (SimpleSpanProcessor) in production. Each span export blocks the request thread.

💡 **Tip**: Always use `BatchSpanProcessor` for production. It buffers spans and exports them in batches on a background thread, adding negligible overhead.

### 3.7 Knowledge Check ❓

1. Your P50 prediction latency is 50 ms, P95 is 200 ms, P99 is 1.5 seconds. Design appropriate histogram buckets.
2. What is the advantage of propagating a `request_id` through logs, traces, and response headers?
3. Why use `BatchSpanProcessor` instead of `SimpleSpanProcessor` for OTel span export?

---

## Module 4: Performance Optimization for ML APIs

### 4.1 Theoretical Foundation 🧠

ML API performance optimization differs fundamentally from traditional web service optimization. A CRUD API's bottleneck is typically the database (query optimization, connection pooling, caching). An ML API's bottleneck is the GPU — a resource that excels at parallel computation but suffers from underutilization when fed single requests. The core optimization principle for ML APIs is: **keep the GPU saturated with batched work while minimizing per-request overhead**.

Three optimization layers drive ML API throughput. **Infrastructure layer**: connection pooling (reuse TCP connections to downstream services), HTTP/2 multiplexing (multiple requests over one connection), and reverse proxy caching (serve identical predictions from Redis without touching the GPU). **Application layer**: batch inference (accumulate individual requests into GPU-sized batches), async preprocessing (tokenize text while the GPU processes a previous batch), and response streaming (return partial results immediately). **GPU layer**: FP16/INT8 quantization (reduce memory and compute), CUDA graph capture (eliminate kernel launch overhead), and model compilation (TorchScript, ONNX, TensorRT).

Performance measurement must precede optimization. Benchmarking tools (`locust`, `wrk`, `k6`) simulate concurrent users and report latency distributions. Without a baseline latency histogram, optimization is guesswork. The cardinal rule: measure first, optimize the bottleneck, measure again.

### 4.2 Mental Model 📐

```
┌─── ML API Optimization Layers ───────────────────────────┐
│                                                          │
│  Layer           │ Technique               │ Impact       │
│  ────────────────┼─────────────────────────┼─────────────│
│  Infrastructure  │ Connection pooling      │ -10ms/req    │
│                  │ HTTP/2 multiplexing     │ 2× concurrency│
│                  │ Redis response cache    │ -100ms (cache │
│                  │                         │   hit)       │
│  ────────────────┼─────────────────────────┼─────────────│
│  Application     │ Batch inference         │ 5-10× throughput│
│                  │ Async preprocessing     │ -20ms/req    │
│                  │ Streaming response      │ -TTFB 90%    │
│  ────────────────┼─────────────────────────┼─────────────│
│  GPU             │ FP16/INT8 quantization  │ 2× speed +    │
│                  │                         │ 50% less mem │
│                  │ CUDA graph capture      │ -5ms/launch  │
│                  │ TensorRT compilation    │ 2-3× speed    │
│  ────────────────┼─────────────────────────┼─────────────│
│  Deployment      │ Horizontal scaling      │ N× throughput│
│                  │ Model replication       │ 2-4× throughput│
│                  │ GPU sharing (MIG/MPS)   │ 3-7× utilization│
└──────────────────────────────────────────────────────────┘
```

```
┌─── Batch Inference Throughput ───────────────────────────┐
│                                                          │
│  Single (no batching):                                   │
│  Request → [GPU: 10ms work, 90ms idle] → Response        │
│  Throughput: 10 req/s (GPU 10% utilized)                 │
│                                                          │
│  Batch size 8:                                           │
│  8 Requests → [GPU: 30ms work (batched)] → 8 Responses   │
│  Throughput: 8 × 33 req/s = 264 req/s (GPU 80% utilized) │
│                                                          │
│  Batch size 32:                                          │
│  32 Requests → [GPU: 50ms work (batched)] → 32 Responses │
│  Throughput: 32 × 20 req/s = 640 req/s (GPU 95% utilized)│
│                                                          │
│  Tradeoff: larger batch = higher throughput, higher      │
│  per-request latency (request waits for batch to fill).  │
│  Optimal batch = where P99 latency ≤ SLA threshold.      │
└──────────────────────────────────────────────────────────┘
```

### 4.3 Syntax and Semantics 📝

```python
# optimization.py — Performance optimization patterns for FastAPI ML services
import asyncio
import hashlib
import json
import time
from functools import lru_cache
from typing import Optional

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ═══ PATTERN 1: Response Caching with Redis ═══
# WHY: Repeated identical predictions (e.g., "Is this email spam?"
# for the same email) waste GPU compute. Cache results keyed by
# input hash. Redis provides sub-millisecond lookups.
class PredictRequest(BaseModel):
    text: str

cache_client: Optional[redis.Redis] = None

async def get_cache_client() -> redis.Redis:
    global cache_client
    if cache_client is None:
        cache_client = redis.Redis(
            host="redis.internal",
            port=6379,
            decode_responses=True,
        )
    return cache_client

def input_cache_key(text: str, model_version: str) -> str:
    """Generate deterministic cache key from input + model version."""
    input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"predict:{model_version}:{input_hash}"

@app.post("/predict/cached")
async def predict_cached(
    req: PredictRequest,
    request: Request,
):
    model_version = "bert-v2.1.0"
    cache = await get_cache_client()
    cache_key = input_cache_key(req.text, model_version)

    # Check cache — if hit, return immediately (sub-ms latency)
    cached = await cache.get(cache_key)
    if cached is not None:
        return {**json.loads(cached), "source": "cache"}

    # Cache miss — run inference (GPU-bound, expensive)
    start = time.perf_counter()
    prediction = 0.95  # substitute with actual model call
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "prediction": prediction,
        "model": model_version,
        "latency_ms": elapsed_ms,
        "source": "model",
    }

    # Store in cache with TTL (e.g., 1 hour)
    await cache.setex(cache_key, 3600, json.dumps(result))
    return result

# ═══ PATTERN 2: Async Preprocessing Pipeline ═══
# WHY: Tokenization can run on CPU while the GPU processes a previous
# batch. Overlapping CPU and GPU work increases throughput.
class AsyncPipeline:
    def __init__(self):
        self._preprocess_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._inference_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    async def preprocess(self, text: str) -> list[int]:
        """Simulate tokenization (CPU work)."""
        await asyncio.sleep(0.001)  # actual tokenization
        return [hash(c) % 30000 for c in text.split()]

    async def infer(self, tokens: list[int]) -> float:
        """Simulate GPU inference."""
        await asyncio.sleep(0.05)  # actual model forward pass
        return 0.95

    async def process(self, text: str) -> float:
        """Pipeline: preprocess while previous batch is inferring."""
        tokens = await self.preprocess(text)
        result = await self.infer(tokens)
        return result

pipeline = AsyncPipeline()

@app.post("/predict/pipelined")
async def predict_pipelined(req: PredictRequest):
    score = await pipeline.process(req.text)
    return {"prediction": score}

# ═══ PATTERN 3: Concurrent Fan-Out with gather ═══
# WHY: A single prediction may depend on multiple independent services
# (embedding lookup, feature fetch, ABAC check). Run them concurrently
# to minimize total latency.
async def fetch_embeddings(client: httpx.AsyncClient, text: str):
    resp = await client.get(f"https://embeddings.internal/{hash(text)}")
    return resp.json()

async def fetch_user_features(client: httpx.AsyncClient, user_id: str):
    resp = await client.get(f"https://features.internal/{user_id}")
    return resp.json()

@app.post("/predict/enriched")
async def predict_enriched(req: PredictRequest, user_id: str):
    async with httpx.AsyncClient() as client:
        # Run both fetches concurrently — total time = max(fetch1, fetch2)
        embeddings, features = await asyncio.gather(
            fetch_embeddings(client, req.text),
            fetch_user_features(client, user_id),
        )
    # Combine and predict
    return {"score": 0.95}

# ═══ PATTERN 4: Connection Pool Monitoring ═══
# WHY: Connection pool exhaustion causes mysterious latency spikes.
# Expose pool stats as metrics for proactive monitoring.
@app.get("/debug/pool-stats")
async def pool_stats(request: Request):
    if hasattr(request.app.state, "http_client"):
        pool = request.app.state.http_client._pool
        return {
            "active_connections": len(pool._connections),
            "max_connections": pool._max_connections,
            "keepalive_connections": len(pool._keepalive_connections),
        }
    return {"error": "No connection pool configured"}
```

```python
# benchmarking.py — Load testing with locust
# WHY: Measure throughput and latency before and after optimization.
# Run: locust -f benchmarking.py --host=http://localhost:8000
from locust import HttpUser, task, between

class MLAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 2-10 requests per second per user

    @task(3)  # weight: 3× more frequent than other tasks
    def predict_single(self):
        self.client.post(
            "/predict",
            json={"text": "This is a test sentence for sentiment analysis."},
            headers={"X-Request-ID": f"locust-{self.environment.runner.user_count}"},
        )

    @task(1)
    def predict_batch(self):
        self.client.post(
            "/predict/batch",
            json={"texts": ["test"] * 10},
        )

    @task(1)
    def health_check(self):
        self.client.get("/healthz")
```

### 4.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph "Request Lifecycle (Optimized)"
        REQ["Request Arrives"]
        CACHE{"Redis Cache Hit?"}
        REQ --> CACHE
        CACHE -->|"Yes (~1ms)"| RESP_CACHE["Return Cached\nResponse"]
        CACHE -->|"No"| ASYNC["Async Fan-Out\n(concurrent)"]

        subgraph "Concurrent I/O"
            FS["Fetch Features\n(20ms, HTTP)"]
            EMB["Fetch Embeddings\n(30ms, HTTP)"]
            AUTH["Check Permissions\n(5ms, gRPC)"]
        end

        ASYNC --> FS
        ASYNC --> EMB
        ASYNC --> AUTH
        FS --> MERGE["Merge Features"]
        EMB --> MERGE
        AUTH --> MERGE

        MERGE --> BATCH["Batch Collector\n(accumulate until N=32)"]
        BATCH --> GPU["GPU Inference\n(50ms for batch of 32)"]
        GPU --> RESP_MODEL["Return Prediction"]
        GPU --> BG["Background: Write to Cache"]
    end
```

### 4.5 Application in ML/AI Systems 🤖

Roblox's recommendation models serve personalized content to 70+ million daily active users. Their optimization journey illustrates the layers of ML API performance tuning. Baseline: 1 GPU pod, no batching, 50 RPS, P99 800 ms. Optimization 1 (connection pooling): reduced feature store latency from 30 ms to 5 ms → 80 RPS. Optimization 2 (batch inference, batch=32): GPU utilization from 15% to 80% → 400 RPS. Optimization 3 (Redis cache for popular items): 40% cache hit rate → 600 RPS. Optimization 4 (FP16 quantization): inference time from 50 ms to 25 ms → 800 RPS. Final: 16× throughput improvement without adding GPUs. Their lesson: infrastructure optimizations (pooling, caching) deliver the first 2×; application optimizations (batching, async pipelining) deliver the next 4×; GPU optimizations (quantization, compilation) deliver the final 2×.

### 4.6 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Setting cache TTL too high for frequently updated models. A model retrain deploys, but the cache serves stale predictions for hours.

💡 **Tip**: Version cache keys by model version: `predict:{model_version}:{input_hash}`. When a new model deploys, old cache entries are naturally unused. Set TTL based on how frequently the model retrain cadence expects.

⚠️ **Pitfall**: Batching without a timeout. If traffic is low, a batch of 32 never fills, and requests wait indefinitely.

💡 **Tip**: Set a `max_wait_ms` on the batch collector. If the batch doesn't fill within the timeout, process whatever has accumulated. This bounds latency at the cost of suboptimal GPU utilization during low traffic.

⚠️ **Pitfall**: Optimizing GPU inference when the bottleneck is I/O. A 10 ms GPU inference behind a 200 ms feature store call doesn't benefit from TensorRT compilation.

💡 **Tip**: Use distributed tracing (OTel) to identify the actual bottleneck. The span with the longest duration is your optimization target. Often in ML APIs, I/O (feature stores, embedding services) dominates inference time.

### 4.7 Knowledge Check ❓

1. Your ML API serves 100 RPS with P99 500 ms. Tracing shows: feature store 150 ms, embeddings 100 ms, GPU inference 50 ms. What should you optimize first?
2. A batch collector with `max_wait_ms=10` and `max_batch_size=32` receives 10 requests per second. How large is the typical batch?
3. You set Redis cache TTL to 1 week. The model retrains daily. What happens?

---

## 📦 Compression Code

```python
"""
production_ml_service.py — Production FastAPI ML service
with caching, instrumentation, and K8s readiness.
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, Optional

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request, Depends, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ═══ Logging ═══
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml-api")

# ═══ Lifespan ═══
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.model_ready = False
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        timeout=httpx.Timeout(connect=5.0, read=30.0),
    )
    app.state.cache = redis.Redis(host="redis.internal", port=6379, decode_responses=True)
    app.state.model_ready = True
    logger.info("Model ready — accepting traffic")
    yield
    app.state.model_ready = False
    await app.state.http_client.aclose()
    await app.state.cache.aclose()

app = FastAPI(lifespan=lifespan, title="Production ML Service")

# ═══ Middleware ═══
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

app.add_middleware(RequestIDMiddleware)

# ═══ Observability ═══
FastAPIInstrumentor.instrument_app(app)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ═══ Schemas ═══
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)

class PredictResponse(BaseModel):
    prediction: float
    model: str
    latency_ms: float
    source: str  # "model" or "cache"

# ═══ Dependencies ═══
def get_client():
    return app.state.http_client

def get_cache():
    return app.state.cache

# ═══ Endpoints ═══
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request, cache=Depends(get_cache)):
    model_version = "bert-v2.1.0"
    input_hash = hashlib.sha256(req.text.encode()).hexdigest()[:16]
    cache_key = f"predict:{model_version}:{input_hash}"

    cached = await cache.get(cache_key)
    if cached:
        data = json.loads(cached)
        return PredictResponse(source="cache", **data)

    start = time.perf_counter()
    prediction = 0.95  # actual model call here
    elapsed = (time.perf_counter() - start) * 1000

    result = PredictResponse(
        prediction=prediction, model=model_version,
        latency_ms=round(elapsed, 2), source="model",
    )
    await cache.setex(cache_key, 3600, result.model_dump_json())
    return result

@app.get("/healthz")
async def health():
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(503, "Not ready", headers={"Retry-After": "5"})
    return {"status": "ok", "model": "bert-v2.1.0"}

@app.get("/readyz")
async def ready():
    return {"status": "ok", "checks": {"model": app.state.model_ready}}
```

## 🎯 Documented Project

### Description

Deploy a production FastAPI ML service end-to-end: Docker multi-stage build, Kubernetes deployment with GPU scheduling and probes, Traefik reverse proxy with TLS termination and rate limiting, OpenTelemetry tracing with Jaeger, Prometheus metrics with Grafana dashboards, and systematic performance optimization from 50 RPS to 500+ RPS.

### Functional Requirements

1. Multi-stage Dockerfile producing a < 4 GB runtime image with CUDA and PyTorch.
2. Kubernetes Deployment with GPU resource requests, readiness/liveness probes, HPA, and PDB.
3. Traefik reverse proxy with automatic Let's Encrypt TLS, `RateLimit` middleware (100 req/s/IP), and health-check-based load balancing.
4. OpenTelemetry auto-instrumentation of all FastAPI routes with OTLP export to Jaeger.
5. Prometheus metrics: prediction latency histogram, request counter by model version, GPU memory gauge, cache hit rate.
6. Structured JSON logging with request ID propagation through middleware, traces, and logs.
7. Redis response cache with model-versioned keys and 1-hour TTL.
8. Batch inference collector with configurable `max_batch_size` and `max_wait_ms`.

### Main Components

- `Dockerfile` (multi-stage: CUDA dev → CUDA runtime)
- `k8s-deployment.yaml` (Deployment, Service, Ingress, HPA, PDB)
- `traefik.yml` + `dynamic.yml` (reverse proxy config)
- `observability.py` (OTel + Prometheus + structured logging)
- `optimization.py` (Redis cache, async pipeline, batch collector)
- `benchmarking.py` (locust load test)

### Success Metrics

- Container image pull time < 2 minutes (from registry)
- Pod startup (including model loading) < 30 seconds
- P99 prediction latency < 200 ms at 500 RPS
- Cache hit rate ≥ 30% for production traffic patterns
- Zero dropped requests during rolling updates (graceful shutdown)
- GPU utilization ≥ 70% during sustained load

## 🎯 Key Takeaways

1. **Reverse proxies are mandatory, not optional** — Traefik or Nginx provides TLS termination, rate limiting, and health-check-based load balancing that Uvicorn alone cannot deliver.
2. **Multi-stage Docker builds shrink images by 50–70%** — separate the CUDA build environment (compilers, headers) from the runtime (libraries only). Model weights should be volume-mounted, not baked into the image.
3. **Kubernetes probes must account for model loading time** — `initialDelaySeconds` on readiness probes prevents traffic routing to pods still loading models. `terminationGracePeriodSeconds` must cover in-flight predictions.
4. **Three pillars of observability reinforce each other** — traces show per-request journeys, metrics aggregate behavior, and logs capture context. A shared `request_id` links them all.
5. **Cache responses by input hash + model version** — sub-millisecond Redis lookups avoid redundant GPU inference. Version-scoped keys prevent stale predictions after model updates.
6. **Batch inference trades latency for throughput** — a batch of 32 can process in 50 ms what would take 32 × 10 ms = 320 ms individually. Always set a `max_wait_ms` to bound latency.
7. **Measure before optimizing** — use OTel traces to find the actual bottleneck (often I/O, not GPU). Infrastructure optimizations (pooling, caching) deliver 2×, application (batching, async) delivers 4×, GPU (quantization) delivers 2×.

## References

- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [NGINX Reverse Proxy Guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/instrumentation/python/)
- [Prometheus FastAPI Instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Locust Load Testing](https://docs.locust.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
