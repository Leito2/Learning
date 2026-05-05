# 🚀 05 - Docker Registry y CI/CD

Construir imágenes Docker es solo la mitad del trabajo. Para un **Backend Engineer**, el flujo profesional implica versionar imágenes, almacenarlas en registries confiables y automatizar su construcción y despliegue mediante pipelines de CI/CD. Para un **ML/AI Engineer**, el registry es el repositorio central de modelos servidos: cada experimento exitoso se convierte en una imagen versionada que puede desplegarse en staging o producción con trazabilidad completa.

Caso real: Un equipo de MLOps en una startup de salud entrena 20 variantes de un modelo de diagnóstico por imágenes a la semana. Cada modelo exitoso se empaqueta como una imagen Docker etiquetada con el hash del commit de Git y la métrica de validación (`model:v1.2.0-auc0.94-a1b2c3d`). Estas imágenes se almacenan en AWS ECR. Cuando un médico reporta un falso negativo, el equipo puede identificar exactamente qué imagen estaba corriendo en producción en esa fecha y reproducir el entorno de inferencia al 100%.

---

## 1. Docker Hub: El Registry Público por Defecto

Docker Hub (`docker.io`) es el registry público predeterminado. Ofrece repositorios públicos gratuitos y privados con límites de rate limiting.

| Característica | Plan Gratuito | Plan Pro |
|----------------|---------------|----------|
| Repositorios privados | 1 | Ilimitados |
| Pulls autenticados | 200 cada 6 horas | 5000 diarios |
| Builds automáticos | Sí | Sí (con más concurrencia) |
| Vulnerability Scanning | Básico | Avanzado (Docker Scout) |

```bash
# Iniciar sesión en Docker Hub
docker login

# Etiquetar una imagen para Docker Hub
docker tag myapp:latest dockerhubusername/myapp:1.0.0

# Subir imagen (push)
docker push dockerhubusername/myapp:1.0.0

# Descargar imagen (pull)
docker pull dockerhubusername/myapp:1.0.0
```

⚠️ **Advertencia**: Docker Hub impone rate limits agresivos para pulls anónimos (100 cada 6 horas por IP). En CI/CD o clusters grandes, esto puede bloquear tus despliegues. Usa siempre autenticación o un registry mirror privado.

---

## 2. Registries Privados: Harbor, ECR, GCR, ACR

En entornos empresariales y de ML/AI, los registries privados son el estándar por razones de seguridad, cumplimiento y control de latencia.

| Registry | Proveedor | Características Destacadas |
|----------|-----------|---------------------------|
| **Harbor** | CNCF Open Source | Escaneo de vulnerabilidades integrado, replicación entre registries, Helm charts, notary para firmas. |
| **AWS ECR** | Amazon Web Services | Integración IAM, soporte OCI, lifecycle policies, image scanning con Clair. |
| **GCP GCR / Artifact Registry** | Google Cloud | Integración con Cloud Build, soporte multi-región, análisis de vulnerabilidades. |
| **Azure ACR** | Microsoft Azure | Tasks para build automático, geo-replicación, integración AKS, Content Trust. |

```bash
# AWS ECR - Login (requiere AWS CLI)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Etiquetar y subir a ECR
docker tag myapp:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:1.0.0
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:1.0.0
```

💡 **Tip**: Configura lifecycle policies en ECR para eliminar automáticamente imágenes antiguas no etiquetadas. Esto evita que el costo de almacenamiento crezca indefinidamente con cada build de CI.

---

## 3. Estrategias de Image Tagging

El versionado de imágenes es crítico para trazabilidad y rollback seguro.

| Estrategia | Formato | Cuándo Usar |
|------------|---------|-------------|
| **SemVer** | `1.2.3` | Releases estables de aplicaciones. |
| **Git SHA** | `a1b2c3d` | Trazabilidad exacta con el código fuente. |
| **SemVer + SHA** | `1.2.3-a1b2c3d` | Balance entre legibilidad y trazabilidad. |
| **Build Number** | `1.2.3-b45` | Integración con pipelines CI (Jenkins, GitLab). |
| **Latest** | `latest` | ⚠️ **Evitar en producción**. Solo para desarrollo local. |

```bash
# Ejemplo de tagging completo
VERSION="1.2.3"
COMMIT_SHA=$(git rev-parse --short HEAD)
BUILD_NUM=${CI_BUILD_NUMBER:-local}

IMAGE="myapp"
docker build -t "${IMAGE}:latest" \
             -t "${IMAGE}:${VERSION}" \
             -t "${IMAGE}:${VERSION}-${COMMIT_SHA}" \
             -t "${IMAGE}:${VERSION}-b${BUILD_NUM}" .

docker push "${IMAGE}" --all-tags
```

⚠️ **Advertencia**: La etiqueta `latest` es mutable y no determinista. Un nodo de Kubernetes que haga `imagePullPolicy: Always` puede descargar una versión diferente de `latest` sin que notes el cambio, causando comportamientos impredecibles en producción. Siempre despliega con tags inmutables.

---

## 4. Docker BuildKit: Builds Modernos y Eficientes

BuildKit es el builder de próxima generación de Docker. Ofrece mejoras significativas en rendimiento y capacidades.

| Característica | Legacy Builder | BuildKit |
|----------------|----------------|----------|
| Paralelismo de capas | Limitado | Completo (DAG de dependencias) |
| Secret mounts | No | Sí (evita exponer secrets en capas) |
| SSH forwarding | No | Sí (para clonar repos privados) |
| Cache externo | No | Sí (cache export/import, registry cache) |
| Multi-platform | No nativo | Sí (`docker buildx`) |

```bash
# Habilitar BuildKit (default en Docker 23.0+)
export DOCKER_BUILDKIT=1

# Build con secret (la clave nunca queda en la imagen)
DOCKER_BUILDKIT=1 docker build --secret id=pip_config,src=$HOME/.pip/pip.conf -t myapp .
```

```dockerfile
# Uso de secret en Dockerfile (BuildKit)
# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=secret,id=pip_config \
    pip install --no-cache-dir -r requirements.txt --config-file=/run/secrets/pip_config
COPY src/ ./src/
CMD ["python", "-m", "src.main"]
```

💡 **Tip**: En CI/CD, usa `--mount=type=cache` para cachear directorios de paquetes (`/root/.cache/pip`, `/root/.npm`) entre builds sin incluirlos en la imagen final.

---

## 5. Buildx y Multi-Arquitectura

`docker buildx` es una extensión de Docker CLI que utiliza BuildKit para construir imágenes multi-plataforma (AMD64, ARM64).

```bash
# Crear un builder multi-plataforma
docker buildx create --name multiarch --use --bootstrap

# Build y push para múltiples arquitecturas
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myapp:1.0.0 \
  --push .
```

Caso real: Un equipo desarrolla una aplicación de edge computing que corre en servidores x86 en la nube y en dispositivos ARM64 (NVIDIA Jetson) en el edge. Con `buildx`, construyen una sola vez y publican una imagen manifest que el runtime Docker descarga automáticamente según la arquitectura del host.

| Arquitectura | Uso típico |
|--------------|------------|
| `linux/amd64` | Servidores cloud (AWS, GCP, Azure), laptops Windows/Mac Intel |
| `linux/arm64` | Apple Silicon (M1/M2), AWS Graviton, Raspberry Pi 4+, NVIDIA Jetson |
| `linux/arm/v7` | Raspberry Pi 3, dispositivos IoT legacy |

⚠️ **Advertencia**: Las builds multi-plataforma pueden ser lentas si no utilizas un builder con emulación QEMU o nodos nativos remotos. Para builds frecuentes en CI, considera usar runners nativos de cada arquitectura.

---

## 6. CI/CD con GitHub Actions

Automatizar el build, test y push de imágenes es fundamental para un flujo profesional.

```yaml
# .github/workflows/docker-build.yml
name: Docker Build and Push

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix=,suffix=,format=short

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

| Paso | Propósito |
|------|-----------|
| `actions/checkout` | Clona el repositorio. |
| `docker/setup-buildx-action` | Configura BuildKit y buildx. |
| `docker/login-action` | Autentica contra el registry (GHCR en este caso). |
| `docker/metadata-action` | Genera tags dinámicos (semver, sha, branch). |
| `docker/build-push-action` | Construye, usa cache de GitHub Actions (`gha`), y hace push. |

💡 **Tip**: El cache `type=gha` almacena capas de build en el cache de GitHub Actions, reduciendo drásticamente el tiempo de build en pipelines subsiguientes.

---

## 7. Optimización de Pipelines CI/CD

### Layer Caching en CI

```yaml
# Ejemplo con cache explícito de capas en GitHub Actions
- uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: myapp:latest
    cache-from: type=registry,ref=myapp:buildcache
    cache-to: type=registry,ref=myapp:buildcache,mode=max
```

### Estrategias de Paralelización

| Estrategia | Descripción |
|------------|-------------|
| **Build matrix** | Construir múltiples variantes (Python 3.10, 3.11, 3.12) en paralelo. |
| **Jobs separados** | Un job para lint/test, otro para build, otro para scan/push. |
| **Conditional builds** | Solo build si cambia el Dockerfile o dependencias (`paths` filter). |

⚠️ **Advertencia**: Almacenar credenciales de registry en variables de entorno planas (`DOCKER_PASSWORD`) puede filtrarse en logs de CI. Usa siempre los mecanismos de secrets nativos de tu plataforma (`secrets.GITHUB_TOKEN`, GitLab CI variables masked, etc.).

---

## 8. Firmas de Imágenes y Content Trust

Para garantizar la integridad y autoría de las imágenes, Docker soporta Docker Content Trust (DCT) basado en Notary.

```bash
# Habilitar Content Trust
export DOCKER_CONTENT_TRUST=1

# Firmar al hacer push
docker trust sign myregistry/myapp:1.0.0

# Verificar firma
docker trust inspect --pretty myregistry/myapp:1.0.0
```

💡 **Tip**: En entornos regulados (healthcare, finance), las firmas de imágenes son obligatorias. Herramientas como Cosign (Sigstore) están ganando tracción como alternativa moderna y más simple a Docker Content Trust.

---

## 9. 📦 Código de Compresión

```bash
# tag-and-push.sh
#!/bin/bash
set -e

REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE="${IMAGE:-myapp}"
VERSION="${1:-$(git describe --tags --always)}"

echo "=== Building ${REGISTRY}/${IMAGE}:${VERSION} ==="

DOCKER_BUILDKIT=1 docker build \
  --build-arg VERSION="$VERSION" \
  --cache-from "${REGISTRY}/${IMAGE}:latest" \
  -t "${REGISTRY}/${IMAGE}:${VERSION}" \
  -t "${REGISTRY}/${IMAGE}:latest" \
  .

echo "=== Pushing ==="
docker push "${REGISTRY}/${IMAGE}:${VERSION}"
docker push "${REGISTRY}/${IMAGE}:latest"

echo "=== Done ==="
```

```yaml
# .github/workflows/docker.yml (versión compacta)
name: Docker CI
on:
  push:
    tags: ["v*"]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

*Imagen: [Docker Hub logo](https://commons.wikimedia.org/wiki/File:Docker_Hub_logo.png)*

*Imagen: [CI/CD pipeline diagram](https://commons.wikimedia.org/wiki/File:CI_CD_Diagram.png)*
