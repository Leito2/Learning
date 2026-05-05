# 🐳 Docker para ML

La containerización es la base de la reproducibilidad en ML/AI Engineering. Docker encapsula no solo el código Python, sino también las librerías de CUDA, drivers de GPU y dependencias del sistema operativo que hacen que un modelo funcione idénticamente en desarrollo, staging y producción.

Sin Docker, el clásico "en mi máquina funciona" se convierte en un cuello de botella operativo que retrasa meses los despliegues de modelos.

---

## 1. Fundamentos del Dockerfile para ML

Un Dockerfile bien diseñado para ML debe cumplir tres objetivos: reproducibilidad, minimización de tamaño y seguridad. La técnica de **multi-stage builds** permite separar la compilación de dependencias del runtime final.

### 1.1 Multi-Stage Build para PyTorch

El siguiente patrón reduce la imagen final descartando herramientas de compilación (gcc, build-essential) que solo se necesitan para instalar librerías con extensiones C++.

```dockerfile
# ---------- STAGE 1: Builder ----------
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip git build-essential \
    && pip install --no-cache-dir --user -r requirements.txt

# ---------- STAGE 2: Runtime ----------
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copiar solo los artefactos instalados desde el builder
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

💡 **Tip:** Usa `--no-cache-dir` en `pip install` para evitar que la caché de paquetes influe la imagen final.

---

## 2. Optimización de Imágenes y .dockerignore

El archivo `.dockerignore` es tan crítico como el `.gitignore`. Evita copiar datasets masivos, checkpoints de entrenamiento y entornos virtuales locales al contexto de build.

```text
# .dockerignore
venv/
__pycache__/
*.pyc
*.pyo
.git/
.github/
data/
notebooks/
*.ipynb
checkpoints/
*.pt
*.ckpt
```

### 2.1 Comparativa de Estrategias de Reducción de Tamaño

| Estrategia | Reducción Aproximada | Complejidad | Recomendación |
|-----------|---------------------|-------------|---------------|
| Multi-stage build | 40-60% | Media | Obligatoria para producción |
| Slim/Runtime base images | 30-50% | Baja | Usar `python:3.11-slim` o runtime CUDA |
| .dockerignore riguroso | 10-30% | Baja | Siempre implementar |
| Minificación con dive | 5-15% | Media | Auditar capas con `wagoodman/dive` |

---

## 3. Soporte GPU y NVIDIA Container Runtime

Para ejecutar contenedores con acceso a GPU, el host debe tener instalados los **NVIDIA Driver**, **NVIDIA Container Toolkit** y configurado el runtime `nvidia` en Docker.

Verificación de disponibilidad GPU dentro del contenedor:

```bash
# Ejecutar con flag --gpus all
docker run --gpus all --rm nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
```

En Docker Compose:

```yaml
services:
  ml-api:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

⚠️ **Advertencia:** No uses imágenes `devel` en producción. Contienen headers de compilación que aumentan la superficie de ataque. La imagen `runtime` es suficiente para inferencia.

---

## 4. Volúmenes para Datos y Modelos

En ML, es una anti-práctica empaquetar datasets o modelos de gran tamaño dentro de la imagen Docker. La solución son los **volúmenes** o **bind mounts**.

### 4.1 Estrategias de Montaje

| Tipo | Uso en ML | Persistencia |
|------|-----------|--------------|
| Bind mount | Desarrollo local, hot-reload de código | Host |
| Named volume | Almacenamiento de modelos descargados | Docker-managed |
| Volume externo (S3/NFS) | Datasets de entrenamiento, checkpoints | Cloud/NAS |

Ejemplo de montaje para desarrollo:

```bash
docker run -v $(pwd)/src:/app/src -v model-cache:/app/models ml-api:latest
```

**Caso real:** OpenAI utiliza volúmenes de red de alto rendimiento (NFS sobre RDMA) para montar checkpoints de modelos de lenguaje masivos en contenedores de inferencia distribuida, evitando duplicar terabytes de pesos en cada imagen.

---

## 5. Docker Compose para Desarrollo Local

Docker Compose orquesta múltiples servicios: la API de inferencia, una base de datos de features, Redis para caché y Prometheus para métricas.

```yaml
version: "3.9"
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./src:/app/src
    environment:
      - MODEL_PATH=/app/models/model.onnx
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
```

💡 **Tip:** Marca los volúmenes de modelos como `:ro` (read-only) para prevenir modificaciones accidentales durante la inferencia.

---

## 6. Registro de Imágenes y Versionado Semántico

Un registro privado (AWS ECR, Google Artifact Registry, Azure ACR o Harbor self-hosted) es indispensable para equipos de ML.

Convención de tags recomendada:

```
registry/project/ml-api:{model-version}-{git-sha}-{stage}
# Ejemplo:
registry/project/ml-api:v2.1.0-a1b2c3d-prod
```

Comandos para CI/CD:

```bash
# Build con cache explícito
DOCKER_BUILDKIT=1 docker build \
  --cache-from=registry/project/ml-api:latest \
  -t registry/project/ml-api:v2.1.0 \
  -t registry/project/ml-api:latest \
  .

docker push registry/project/ml-api:v2.1.0
docker push registry/project/ml-api:latest
```

---

## 7. Seguridad en Contenedores ML

La seguridad en contenedores de ML abarca tres capas:

1. **Imagen:** Escanea vulnerabilidades con Trivy o Snyk. Las imágenes oficiales de Python a menudo contienen CVEs de alto severidad en OpenSSL o glibc.
2. **Runtime:** Ejecuta contenedores con usuario no-root (`USER 1000`).
3. **Red:** Aísla redes con Docker networks internas; no expongas puertos de debug (5000, 8080) innecesariamente.

```dockerfile
# Crear usuario no-root
RUN groupadd -r mluser && useradd -r -g mluser mluser
USER mluser
```

⚠️ **Advertencia:** Nunca incluyas credenciales de API (AWS keys, tokens de wandb) en el Dockerfile. Usa Docker secrets o variables de entorno inyectadas en runtime.

---

## 8. Comparativa CPU vs GPU Containers

| Métrica | Container CPU | Container GPU |
|---------|---------------|---------------|
| Base image | `python:3.11-slim` | `nvidia/cuda:12.1-runtime-ubuntu22.04` |
| Tamaño base | ~150 MB | ~2.5 GB |
| Tiempo de arranque | < 1 s | 3-5 s (carga CUDA) |
| Throughput inferencia | 10-50 req/s | 500-2000 req/s |
| Costo por hora | Bajo | 5x-10x mayor |
| Runtime flag | Ninguno | `--gpus all` o `runtime: nvidia` |
| Ideal para | Pre/post-procesamiento, modelos ligeros | Deep Learning (CNN, Transformer) |

---

## 📦 Código de Compresión

```bash
# Estructura completa del proyecto Docker para ML
ml-docker-project/
├── Dockerfile                 # Multi-stage optimizado
├── .dockerignore             # Exclusiones críticas
├── docker-compose.yml        # Dev stack completo
├── docker-compose.prod.yml   # Override de producción
├── requirements.txt          # Dependencias Python
├── src/
│   ├── main.py               # FastAPI app
│   └── inference.py          # Wrapper del modelo
├── models/
│   └── model.onnx            # Artefacto versionado (no en imagen)
└── scripts/
    └── build_push.sh         # CI helper
```
