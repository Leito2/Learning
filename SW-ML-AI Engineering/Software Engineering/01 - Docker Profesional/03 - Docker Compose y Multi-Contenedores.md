# 🎼 03 - Docker Compose y Multi-Contenedores

Las aplicaciones modernas rara vez consisten en un solo contenedor. Un backend típico requiere una base de datos, un cache, un message broker y a veces un worker. Para un **ML/AI Engineer**, el stack puede incluir un servicio de inferencia (FastAPI), una base de datos vectorial (PostgreSQL + pgvector), un cache de resultados (Redis) y un job scheduler. Docker Compose es la herramienta estándar para orquestar estos servicios en desarrollo y en entornos de staging ligeros.

Caso real: Un equipo de MLOps desarrolla un sistema de RAG (Retrieval Augmented Generation). Necesitan coordinar: (1) una API de embeddings con SentenceTransformers, (2) PostgreSQL con extensión pgvector para store de vectores, (3) Redis para cachear queries frecuentes, y (4) un servicio de streaming con LangChain. Sin Compose, cada desarrollador debía iniciar 4 contenedores manualmente con redes y volúmenes consistentes. Con Compose, un solo `docker compose up` levanta todo el ecosistema.

---

## 1. Sintaxis de docker-compose.yml

El archivo `docker-compose.yml` define servicios, redes y volúmenes en formato YAML.

```yaml
version: "3.9"

services:
  web:
    build: ./web
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db
      - redis
    networks:
      - backend

  db:
    image: postgres:15-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    networks:
      - backend

  redis:
    image: redis:7-alpine
    networks:
      - backend

volumes:
  pgdata:

networks:
  backend:
    driver: bridge
```

| Clave | Descripción |
|-------|-------------|
| `version` | Versión del formato Compose (3.9+ recomendado para nuevas funciones). |
| `services` | Define los contenedores de la aplicación. Cada servicio puede escalar independientemente. |
| `networks` | Redes Docker personalizadas para comunicación entre servicios. |
| `volumes` | Volúmenes nombrados para persistencia de datos. |

⚠️ **Advertencia**: La clave `version` en Compose es diferente de la versión de Docker Engine. Compose file format 3.9 requiere Docker Engine 19.03.0+.

---

## 2. Comunicación entre Servicios y DNS Interno

Compose crea automáticamente una red bridge y configura un **DNS interno**. Los contenedores pueden comunicarse usando el nombre del servicio como hostname.

```mermaid
graph TB
    subgraph "Red backend (bridge)"
        W[web<br/>8000]
        D[db<br/>5432]
        R[redis<br/>6379]
    end
    W -->|postgres://user:pass@db:5432| D
    W -->|redis://redis:6379| R
```

💡 **Tip**: El DNS interno de Compose resuelve nombres de servicio a IPs dentro de la red. Si escalas un servicio (`docker compose up -d --scale worker=3`), el DNS round-robin distribuye las conexiones entre las réplicas.

---

## 3. depends_on y Orden de Inicio

`depends_on` controla el **orden de creación e inicio**, pero **no espera a que el servicio esté listo**.

```yaml
services:
  api:
    build: ./api
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
```

| Condición | Comportamiento |
|-----------|----------------|
| `service_started` | Espera a que el contenedor esté iniciado (default). No garantiza que la app esté lista. |
| `service_healthy` | Espera a que el healthcheck del servicio pase. Requiere definir `healthcheck` en el servicio dependiente. |
| `service_completed_successfully` | Espera a que el contenedor termine con exit code 0. Útil para migraciones. |

⚠️ **Advertencia**: `depends_on` no espera que PostgreSQL haya terminado de inicializar su data directory. Si tu aplicación intenta conectarse inmediatamente, fallará. Implementa retry con backoff en tu código o usa `condition: service_healthy`.

---

## 4. Variables de Entorno y Archivos .env

Compose soporta múltiples métodos para inyectar configuración:

| Método | Uso | Prioridad |
|--------|-----|-----------|
| Archivo `.env` | Variables para interpolación en `docker-compose.yml` | Baja |
| `environment:` | Define variables directamente en el servicio | Media |
| `env_file:` | Carga variables desde un archivo específico | Media-Alta |
| Variables de shell del host | `export DATABASE_URL=...` | Alta |
| `--env-file` flag | Especifica un archivo de entorno al ejecutar | Máxima |

```yaml
# docker-compose.yml
services:
  api:
    build: ./api
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - DEBUG=${DEBUG:-false}
    env_file:
      - .env.api
```

```bash
# .env
API_PORT=8000
DATABASE_URL=postgres://user:pass@db:5432/mydb
DEBUG=true
```

💡 **Tip**: Nunca comitees archivos `.env` con credenciales. Añádelos a `.gitignore` y proporciona un `.env.example` con valores dummy para que los desarrolladores sepan qué variables configurar.

---

## 5. Perfiles (Profiles) y Entornos Específicos

Los **profiles** permiten definir servicios opcionales que solo se inician cuando se solicitan explícitamente.

```yaml
services:
  api:
    build: ./api
    profiles: [""]

  pgadmin:
    image: dpage/pgadmin4:latest
    profiles: ["debug"]
    ports:
      - "5050:80"

  test-runner:
    build:
      context: ./api
      target: test
    profiles: ["test"]
    command: pytest
```

```bash
# Solo servicios base
docker compose up -d

# Incluir herramientas de debug
docker compose --profile debug up -d

# Ejecutar tests
docker compose --profile test up --abort-on-container-exit
```

Caso real: El equipo de ML usa un servicio `jupyter` con GPU solo para experimentación local. Marcándolo con `profile: ["dev"]`, los servidores de CI nunca lo inician accidentalmente, ahorrando recursos.

---

## 6. Escalado y Políticas de Reinicio

### Escalado horizontal

```bash
# Escalar el servicio worker a 3 réplicas
docker compose up -d --scale worker=3
```

### Políticas de reinicio

| Política | Descripción |
|----------|-------------|
| `no` | Nunca reinicia (default). |
| `always` | Reinicia siempre que el contenedor se detenga. |
| `unless-stopped` | Reinicia siempre excepto si el usuario lo detuvo manualmente. |
| `on-failure` | Reinicia solo si el contenedor termina con un error (exit code != 0). |

```yaml
services:
  worker:
    build: ./worker
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

⚠️ **Advertencia**: `restart: always` en desarrollo puede ser frustrante porque un contenedor con un bug de configuración se reiniciará infinitamente, consumiendo CPU del host en loops de crash. Usa `unless-stopped` para desarrollo local.

---

## 7. Healthchecks en Compose

Definir healthchecks a nivel de Compose permite que `depends_on` con `condition: service_healthy` funcione correctamente.

```yaml
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  api:
    build: ./api
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 8. Override Files y Entornos

Compose permite múltiples archivos YAML que se fusionan. El patrón estándar es:

- `docker-compose.yml` - Configuración base compartida.
- `docker-compose.override.yml` - Configuración de desarrollo (montajes bind, puertos expuestos, debug).
- `docker-compose.prod.yml` - Configuración de producción (replicas, healthchecks, recursos).

```yaml
# docker-compose.override.yml (desarrollo)
version: "3.9"
services:
  api:
    volumes:
      - ./api/src:/app/src
    environment:
      - DEBUG=true
    command: uvicorn src.main:app --reload --host 0.0.0.0
```

```bash
# Desarrollo (carga automáticamente override si existe)
docker compose up -d

# Producción (especificar archivo explícitamente)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

💡 **Tip**: El archivo `docker-compose.override.yml` se carga automáticamente si existe junto a `docker-compose.yml`. Úsalo para configuraciones locales que no quieres que afecten producción.

---

## 9. Comandos Esenciales de Docker Compose

| Comando | Descripción |
|---------|-------------|
| `docker compose up -d` | Crea e inicia contenedores en segundo plano. |
| `docker compose down` | Detiene y elimina contenedores, redes y volúmenes (opcional). |
| `docker compose ps` | Lista contenedores del proyecto. |
| `docker compose logs -f` | Sigue logs de todos los servicios. |
| `docker compose exec <service> <cmd>` | Ejecuta un comando en un servicio en ejecución. |
| `docker compose build` | Construye o reconstruye imágenes. |
| `docker compose pull` | Descarga imágenes actualizadas del registry. |
| `docker compose config` | Valida y muestra la configuración final fusionada. |

```bash
# Ver configuración final resuelta (útil para debug)
docker compose -f docker-compose.yml -f docker-compose.prod.yml config

# Ejecutar migraciones en el servicio api
docker compose exec api alembic upgrade head

# Ver logs solo de db y redis
docker compose logs -f db redis
```

---

## 10. Caso Real: Aplicación con App + DB + Cache

Caso real: Una plataforma de predicción de churn para e-commerce usa FastAPI, PostgreSQL y Redis. La API cachea resultados de modelos scikit-learn en Redis para reducir la latencia de inferencia de 200ms a 5ms en consultas repetidas.

```yaml
version: "3.9"

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/churn_db
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=/app/models/churn_model.pkl
    volumes:
      - ./models:/app/models:ro
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - app_network
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: churn_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d churn_db"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    networks:
      - app_network
    restart: unless-stopped

volumes:
  postgres_data:

networks:
  app_network:
    driver: bridge
```

⚠️ **Advertencia**: En producción, nunca expongas PostgreSQL o Redis directamente al host. El mapeo de puertos en `db` y `redis` debe omitirse o restringirse a `127.0.0.1` si es absolutamente necesario.

---

## 11. 📦 Código de Compresión

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: ./app
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    networks:
      - backend

  db:
    image: postgres:15-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASS}
      POSTGRES_DB: ${DB_NAME}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      retries: 5
    networks:
      - backend

  cache:
    image: redis:7-alpine
    networks:
      - backend

volumes:
  pgdata:

networks:
  backend:
    driver: bridge
```

```bash
# up.sh
#!/bin/bash
set -e
docker compose down
docker compose build
docker compose up -d
docker compose ps
```

```bash
# down.sh
#!/bin/bash
docker compose down -v
```
