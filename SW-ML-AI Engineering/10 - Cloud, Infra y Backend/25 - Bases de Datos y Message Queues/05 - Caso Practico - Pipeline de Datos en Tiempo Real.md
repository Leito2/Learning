# 🚀 Caso Práctico: Pipeline de Datos en Tiempo Real

Este caso práctico integra todo lo aprendido en el módulo en un proyecto completo: un **pipeline ETL en tiempo real para un sistema de recomendación**. El objetivo es capturar eventos de usuario, enriquecerlos con datos transaccionales y de perfil, generar recomendaciones y servirlas con latencia mínima. Este tipo de arquitectura es el estándar en empresas como Netflix, Spotify y Amazon para personalización en tiempo real.


## 1. Requisitos del Sistema

### 1.1 Funcionales

- **Ingestión de eventos:** Capturar clicks, vistas, compras y ratings de usuarios en tiempo real.
- **Almacenamiento transaccional:** Mantener catálogo de productos, precios e inventario con consistencia fuerte.
- **Perfiles de usuario:** Almacenar historial, preferencias y embeddings de usuario en esquema flexible.
- **Caché de recomendaciones:** Servir top-N recomendaciones con latencia sub-50ms.
- **Procesamiento:** Actualizar modelos de recomendación incrementalmente a medida que llegan eventos.

### 1.2 No Funcionales

| Métrica | Objetivo | Fórmula |
|---------|----------|---------|
| **Throughput** | > 50,000 eventos/segundo | $$\lambda = \frac{N_{\text{eventos}}}{\Delta t}$$ |
| **Latencia (p99)** | < 50ms para lectura de recomendaciones | $$L = t_{\text{respuesta}} - t_{\text{request}}$$ |
| **Data Freshness** | < 5 segundos desde evento hasta recomendación actualizada | $$\delta = t_{\text{now}} - t_{\text{last\_update}}$$ |
| **Disponibilidad** | 99.9% uptime | $$A = \frac{T_{\text{up}}}{T_{\text{up}} + T_{\text{down}}} \times 100$$ |

⚠️ **Advertencia:** La latencia de red entre el data center de ingestión y el de servicio puede dominar la métrica de *freshness*. Coloca los consumidores de Kafka lo más cerca posible de la base de datos de escritura.


## 2. Arquitectura del Pipeline

El sistema utiliza una arquitectura Lambda simplificada (solo la capa de velocidad, dado que el batch histórico ya existe en el data lake).

```mermaid
graph LR
    A[Usuarios] -->|Eventos| B[Kafka<br/>Topic: user-events]
    B --> C[Consumer Python]
    C --> D[(PostgreSQL<br/>Catálogo + Transacciones)]
    C --> E[(MongoDB<br/>Perfiles de Usuario)]
    C --> F[(Redis<br/>Cache de Recomendaciones)]
    F --> G[API de Recomendaciones]
    G --> A
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
```

### 2.1 Flujo de Datos

1. **Ingestión:** La aplicación web/móvil publica eventos en formato Avro al topic `user-events` de Kafka.
2. **Procesamiento:** Un consumer en Python (grupo `recommendation-updater`) lee los eventos.
3. **Enriquecimiento:** El consumer consulta PostgreSQL para validar el producto y MongoDB para obtener el perfil del usuario.
4. **Predicción:** Un modelo ligero (o una heurística basada en contenido) genera nuevas recomendaciones.
5. **Actualización:** Las recomendaciones se escriben en Redis con TTL de 1 hora.
6. **Servicio:** La API de recomendaciones lee directamente de Redis.

**Caso real:** Un marketplace de comercio electrónico implementó una arquitectura similar para actualizar recomendaciones de "frecuentemente comprados juntos" en menos de 3 segundos tras una compra, incrementando la tasa de conversión en un 12%.


## 3. Implementación del Consumer en Python

El consumer es el corazón del pipeline. Debe ser tolerante a fallos, idempotente y eficiente en el uso de conexiones.

```python
import json
import logging
from datetime import datetime

import redis
from kafka import KafkaConsumer
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommendation-consumer")

# Conexiones
consumer = KafkaConsumer(
    "user-events",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    group_id="recommendation-updater",
    auto_offset_reset="earliest",
    enable_auto_commit=False,
)

pg_conn = psycopg2.connect(
    host="localhost", database="ecommerce", user="admin", password="secret"
)

mongo_client = MongoClient("mongodb://localhost:27017/")
user_profiles = mongo_client["ml_platform"]["user_profiles"]

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def fetch_product_from_postgres(product_id: str):
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, name, category, price FROM products WHERE id = %s",
            (product_id,),
        )
        return cur.fetchone()


def fetch_user_profile(user_id: str):
    return user_profiles.find_one({"user_id": user_id})


def generate_recommendations(user_id: str, event_type: str, product: dict, profile: dict):
    """Heurística simplificada de recomendación basada en categoría."""
    # En producción, aquí invocarías un modelo de ML
    category = product["category"]
    recommendations = [f"{category}_item_{i}" for i in range(1, 6)]
    return recommendations


def process_message(msg):
    user_id = msg["user_id"]
    product_id = msg["product_id"]
    event_type = msg["event_type"]
    event_offset = msg.get("event_offset", 0)

    # Idempotencia: verificar último offset procesado
    last_offset_key = f"last_offset:{user_id}"
    last_offset = redis_client.get(last_offset_key)
    if last_offset and int(last_offset) >= event_offset:
        logger.info(f"Evento {event_offset} ya procesado. Ignorando.")
        return

    # Enriquecimiento
    product = fetch_product_from_postgres(product_id)
    if not product:
        logger.warning(f"Producto {product_id} no encontrado.")
        return

    profile = fetch_user_profile(user_id)
    if not profile:
        logger.info(f"Creando perfil vacío para usuario {user_id}")
        profile = {"user_id": user_id, "history": []}
        user_profiles.insert_one(profile)

    # Generar y cachear recomendaciones
    recs = generate_recommendations(user_id, event_type, product, profile)
    redis_client.setex(f"recs:{user_id}", 3600, json.dumps(recs))
    redis_client.set(last_offset_key, event_offset)

    # Actualizar perfil en MongoDB
    user_profiles.update_one(
        {"user_id": user_id},
        {"$push": {"history": {"product_id": product_id, "event": event_type, "ts": datetime.utcnow().isoformat()}}}
    )

    logger.info(f"Recomendaciones actualizadas para usuario {user_id}")


def main():
    logger.info("Consumer iniciado. Esperando eventos...")
    for message in consumer:
        try:
            process_message(message.value)
            consumer.commit_sync()
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            # En producción, enviar a DLQ tras N reintentos


if __name__ == "__main__":
    main()
```

💡 **Tip:** Usa `enable_auto_commit=False` y `commit_sync()` manualmente después de un procesamiento exitoso. Esto garantiza que no pierdas mensajes ni los reproceses innecesariamente ante fallos parciales.


## 4. Métricas y Monitoreo

Para garantizar que el pipeline cumple los SLAs, debes exponer y alertar sobre las siguientes métricas:

| Métrica | Instrumentación | Umbral de Alerta |
|---------|-----------------|------------------|
| Consumer Lag | `kafka-consumer-groups.sh --describe` | > 10,000 offsets |
| Redis Hit Rate | `INFO stats` -> `keyspace_hits / (hits + misses)` | < 85% |
| PostgreSQL Query Time | `pg_stat_statements` | p99 > 100ms |
| End-to-End Latency | Timestamp tracing en headers de mensaje | p99 > 5s |

⚠️ **Advertencia:** Un consumer lag creciente indica que tus consumidores no pueden mantener el ritmo de producción. Soluciones: escalar consumidores (hasta igualar particiones), optimizar queries de enriquecimiento o reducir la lógica de procesamiento por mensaje.


## 5. 🎯 Proyecto Documentado

A continuación se describe la estructura completa del proyecto para que puedas desplegarlo localmente con Docker Compose.

### 5.1 Estructura de Carpetas

- `realtime-rec-pipeline/`
  - `docker-compose.yml`
  - `consumer/`
    - `Dockerfile`
    - `requirements.txt`
    - `main.py`
  - `init-scripts/`
    - `init_postgres.sql`
    - `init_mongo.js`
  - `api/`
    - `Dockerfile`
    - `requirements.txt`
    - `app.py`
  - `README.md`

### 5.2 Docker Compose

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: ecommerce
    ports:
      - "5432:5432"
    volumes:
      - ./init-scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - ./init-scripts/init_mongo.js:/docker-entrypoint-initdb.d/init.js

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  consumer:
    build: ./consumer
    depends_on:
      - kafka
      - postgres
      - mongodb
      - redis
    environment:
      KAFKA_BROKERS: kafka:9092
      POSTGRES_DSN: "postgresql://admin:secret@postgres:5432/ecommerce"
      MONGO_URI: "mongodb://mongodb:27017/ml_platform"
      REDIS_HOST: redis
```

### 5.3 Inicialización de PostgreSQL

```sql
-- init_postgres.sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO products (name, category, price) VALUES
('Laptop Pro', 'electronics', 1299.99),
('Auriculares NoiseCancel', 'electronics', 249.50),
('Zapatillas Runner', 'sports', 89.99);
```

### 5.4 API de Recomendaciones (app.py)

```python
from flask import Flask, jsonify
import redis
import json

app = Flask(__name__)
r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

@app.route('/recommendations/<user_id>')
def get_recommendations(user_id: str):
    raw = r.get(f"recs:{user_id}")
    if not raw:
        return jsonify({"user_id": user_id, "recommendations": []}), 404
    return jsonify({"user_id": user_id, "recommendations": json.loads(raw)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 5.5 Tests de Integración

```python
# test_pipeline.py (pytest)
import requests
import json
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def test_end_to_end_recommendation():
    event = {"user_id": "u_test_1", "product_id": 1, "event_type": "click", "event_offset": 999}
    producer.send("user-events", event)
    producer.flush()
    
    # Esperar procesamiento
    import time; time.sleep(2)
    
    resp = requests.get("http://localhost:5000/recommendations/u_test_1")
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) > 0
```


![Diagrama de pipeline de datos](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Data_pipeline.svg/640px-Data_pipeline.svg.png)

*Figura: Concepto de pipeline de datos. Fuente: Wikimedia Commons.*


## 📦 Código de Compresión

Script para empaquetar el proyecto completo del caso práctico:

```python
import os
import zipfile
from pathlib import Path

PROJECT_DIR = Path("realtime-rec-pipeline")
OUTPUT_ZIP = Path("realtime-rec-pipeline.zip")

def comprimir_proyecto():
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Excluir __pycache__ y .git
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'venv'}]
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(PROJECT_DIR.parent if PROJECT_DIR.parent != Path('.') else Path('.'))
                zf.write(file_path, arcname)
                print(f"Agregado: {arcname}")
    print(f"\n✅ Proyecto comprimido en: {OUTPUT_ZIP}")

if __name__ == "__main__":
    comprimir_proyecto()
```
