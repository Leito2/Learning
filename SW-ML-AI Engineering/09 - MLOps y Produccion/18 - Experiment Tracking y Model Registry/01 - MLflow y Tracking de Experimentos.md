# 📊 MLflow y Tracking de Experimentos

## Introduction

En ingeniería de ML, la experimentación no es un evento único sino un proceso iterativo y colaborativo. Sin un sistema de tracking riguroso, los equipos pierden visibilidad sobre qué configuraciones produjeron qué resultados, generando "deuda técnica oculta" y dificultando la reproducibilidad. MLflow Tracking resuelve este problema proporcionando una API unificada para registrar parámetros, métricas, artefactos y dependencias de cada ejecución.

MLflow es un proyecto de la Linux Foundation respaldado por Databricks y adoptado masivamente en la industria. Su arquitectura modular cubre cuatro dominios: Tracking, Model Registry, Projects y Pipelines. Dominar MLflow Tracking es el primer paso hacia pipelines de ML reproducibles, versionados y listos para producción. Para profundizar en el Model Registry, consulta [[03 - Model Registry y Lifecycle|🏛️ Model Registry y Lifecycle]].

---

## 1. 🧠 Fundamentos Teóricos del Experiment Tracking

El tracking de experimentos responde a tres preguntas fundamentales:

- **¿Qué se ejecutó?** (código, versión de datos, entorno)
- **¿Con qué configuración?** (hiperparámetros, preprocesamiento)
- **¿Qué resultado se obtuvo?** (métricas, modelos, logs)

La reproducibilidad exige que cualquier run $R$ pueda ser reconstruido. Formalmente, dado un run $R$, buscamos:

$$
\text{Reproducible}(R) \iff \text{Code}_R \times \text{Data}_R \times \text{Env}_R \times \text{Params}_R \rightarrow \text{Metrics}_R
$$

Donde $\times$ denota el producto cartesiano de factores controlados.

### El problema de la deuda técnica oculta

Sin tracking, los equipos acumulan cuatro tipos de deuda:

| Tipo de Deuda | Síntoma | Consecuencia |
|---|---|---|
| **Configuración** | "¿Qué learning rate usé hace 3 semanas?" | Pérdida del mejor modelo |
| **Datos** | "No sé qué versión del dataset usé" | Imposibilidad de reproducir |
| **Entorno** | "En mi máquina sí funciona" | Inconsistencia entre dev/prod |
| **Métricas** | "La accuracy era buena pero no anoté el F1" | Evaluación incompleta |

El resultado es lo que Sculley et al. llaman "CACE: Changing Anything Changes Everything". MLflow combate cada una de estas deudas con mecanismos de registro explícitos.

---

## 2. 📐 Modelo Mental

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTO                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Run A     │  │  Run B     │  │  Run C     │  ... Run N    │
│  │  ────────  │  │  ────────  │  │  ────────  │               │
│  │  Params:   │  │  Params:   │  │  Params:   │               │
│  │  ├─lr=0.01 │  │  ├─lr=0.001│  │  ├─lr=0.1  │               │
│  │  ├─epochs= │  │  ├─epochs= │  │  ├─epochs= │               │
│  │  │  50     │  │  │  100    │  │  │  200    │               │
│  │  Metrics:  │  │  Metrics:  │  │  Metrics:  │               │
│  │  ├─acc=0.82│  │  ├─acc=0.87│  │  ├─acc=0.78│               │
│  │  ├─f1=0.79 │  │  ├─f1=0.85 │  │  ├─f1=0.74 │               │
│  │  Artifacts:│  │  Artifacts:│  │  Artifacts:│               │
│  │  ├─model   │  │  ├─model   │  │  ├─model   │               │
│  │  ├─plot.png│  │  ├─cm.png  │  │  ├─plot.png│               │
│  │  Tags:     │  │  Tags:     │  │  Tags:     │               │
│  │  ├─git:abc │  │  ├─git:def │  │  ├─git:ghi │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

Cada run es una tupla inmutable `R = (params, metrics, artifacts, tags)`. El scheduler de HPO o el científico de datos crea runs, y la UI permite compararlos, filtrarlos y seleccionar el mejor para promoción al Model Registry.

---

## 3. 🏗️ Arquitectura de MLflow

MLflow se compone de cuatro pilares:

| Componente | Descripción | Backend |
|---|---|---|
| **Tracking API** | Interfaces de Python, Java, R, REST para loggear información | HTTP/REST |
| **Backend Store** | Almacena metadatos (params, metrics, tags) | SQLAlchemy: SQLite, PostgreSQL, MySQL |
| **Artifact Store** | Almacena archivos grandes (modelos, plots, datasets) | Filesystem local, S3, GCS, Azure Blob, HDFS |
| **MLflow UI** | Interfaz web para visualizar, comparar y filtrar runs | React SPA sobre Flask/Gunicorn |

```mermaid
graph TB
    subgraph "Cliente de Entrenamiento"
        A[Script Python] -->|mlflow.log_param/log_metric| B[(Tracking API)]
        A -->|mlflow.log_artifact/log_model| C[(Artifact Store)]
    end
    subgraph "Servidor MLflow"
        B --> D[(Backend Store<br/>PostgreSQL)]
        E[MLflow UI] --> D
        E --> C
    end
    subgraph "Almacenamiento"
        C --> F[AWS S3]
        C --> G[Azure Blob]
        C --> H[GCS]
        C --> I[Local FS]
    end
```

![Arquitectura de flujo de datos](https://upload.wikimedia.org/wikipedia/commons/0/0d/Data-flow-diagram-example.svg)

### Estrategias de despliegue

| Estrategia | Escenario | Tracking URI | Backend Store | Artifact Store |
|---|---|---|---|---|
| **Local** | Desarrollo individual | `file:///tmp/mlruns` | SQLite local | Filesystem local |
| **Remoto sin proxy** | Equipo pequeño | `http://server:5000` | PostgreSQL remoto | S3/GCS bucket |
| **Remoto con proxy** | Empresa, multi-team | `https://mlflow.company.com` | PostgreSQL+Proxy | S3 presigned URLs |
| **Databricks Managed** | Enterprise, cloud-native | Automático | Unity Catalog | DBFS/S3 automático |

---

## 4. 💻 MLflow Tracking en la Práctica

### Pipeline Tracked Completo

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Configurar tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run(run_name="rf_gridsearch_v2"):
        # --- Tags: metadata inmutables ---
        mlflow.set_tag("data_version", "v1.0")
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("developer", "ml_team")
        mlflow.set_tag("git_commit", "a1b2c3d")

        # --- Parámetros ---
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "criterion": "gini",
            "random_state": 42
        }
        mlflow.log_params(params)

        # --- Carga y split ---
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Entrenamiento ---
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        # --- Métricas ---
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # --- Artefactos: gráficos ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        fig.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # --- Artefacto: modelo ---
        signature = mlflow.models.infer_signature(
            X_test, clf.predict(X_test)
        )
        mlflow.sklearn.log_model(
            clf, "model",
            signature=signature,
            input_example=X_test[:2]
        )

        print(f"Run completado: acc={acc:.4f}, f1={f1:.4f}")

if __name__ == "__main__":
    main()
```

### Ciclo de vida de un Run

```mermaid
stateDiagram-v2
    [*] --> Running: start_run()
    Running --> Running: log_param/log_metric
    Running --> Running: log_artifact
    Running --> Finished: end_run() implícito
    Running --> Failed: excepción no capturada
    Finished --> [*]
    Failed --> [*]
```

---

## 5. 🔄 MLflow Pipelines y Projects

MLflow ofrece dos mecanismos para estandarizar y empaquetar flujos de trabajo:

### MLflow Projects

Un MLflow Project es un directorio con un archivo `MLproject` (YAML) que define:
- Entorno (conda, docker)
- Entry points (comandos ejecutables)
- Parámetros esperados

```yaml
# MLproject
name: iris_classifier
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 200}
      max_depth: {type: int, default: 10}
    command: "python train.py --n_estimators {n_estimators} --max_depth {max_depth}"
```

Ejecución desde cualquier entorno:

```bash
mlflow run git@github.com:equipo/iris-project.git -P n_estimators=300
```

### MLflow Pipelines (antes MLflow Recipes)

Template-based workflows para tareas comunes (regression, classification). Un pipeline hereda de un template y define steps en `pipeline.yaml`:

```mermaid
graph LR
    A[Ingest] --> B[Split]
    B --> C[Transform]
    C --> D[Train]
    D --> E[Evaluate]
    E --> F[Register]
    F --> C
    style A fill:#ccffcc
    style F fill:#ffcccc
```

Cada step produce artefactos trackeados automáticamente. Los pipelines son la evolución productiva del tracking manual: reemplazan scripts ad-hoc por templates mantenibles con auto-logging y validación de schemas.

---

## 6. 📊 MLflow UI y Visualización

La UI de MLflow permite:

- **Comparación multi-run:** selecciona runs por checkbox y compáralos lado a lado en tablas y gráficos.
- **Parallel coordinates:** visualiza relaciones param→metric en hiperespacios de alta dimensionalidad.
- **Scatter plots:** cualquier param vs cualquier metric como scatter interactivo.
- **Contour plots:** regiones de máxima performance en el espacio de hiperparámetros.

Acceso típico: `http://localhost:5000`

### API de Consulta para Visualización Programática

```python
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()
experiment = client.get_experiment_by_name("iris_classification")

# Obtener todos los runs con filter
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.accuracy DESC"],
    max_results=20
)

# DataFrame para análisis
df = pd.DataFrame([
    {
        "run_id": r.info.run_id,
        "run_name": r.data.tags.get("mlflow.runName", ""),
        **{f"param.{k}": v for k, v in r.data.params.items()},
        **{f"metric.{k}": v for k, v in r.data.metrics.items()},
    }
    for r in runs
])

print(f"Top runs por accuracy:\n{df.head()}")
```

---

## 7. ⚡ Autologging Avanzado

MLflow proporciona autologging para frameworks populares. La fórmula general:

$$
\text{Autolog}(F) \rightarrow \forall \text{ step } s \in \text{training\_loop}, \text{record}(\text{params}_s, \text{metrics}_s, \text{model}_s)
$$

```python
import mlflow

# Una línea reemplaza todo el boilerplate de logging
mlflow.sklearn.autolog()
mlflow.tensorflow.autolog(log_models=True)
mlflow.pytorch.autolog()
mlflow.keras.autolog()
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()
mlflow.spark.autolog()

# Para transformers/HuggingFace (requiere mlflow[transformers])
mlflow.transformers.autolog()
```

⚠️ **Advertencia con autologging:**

1. **Parámetros sensibles:** Autologging captura TODOS los hiperparámetros, incluyendo rutas de archivos y credenciales si las pasaste como argumentos.
2. **Control granular:** En pipelines productivos, desactiva autologging global y registra manualmente. Autologging es ideal para exploración, no para producción.
3. **Overhead I/O:** Autologging serializa el modelo en cada run. Para entrenamientos lentos, esto es negligible, pero para HPO rápido puede añadir latencia significativa.

---

## 8. 📈 Logging de Métricas Temporales

Para entrenamientos largos (deep learning, RL), registra métricas por epoch con `step`:

```python
import mlflow
import torch
import torch.nn as nn

def train(model, loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", "Adam")

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Logging por step para series temporales
        mlflow.log_metric("train_loss", train_loss / len(loader), step=epoch)

        # Validación
        val_acc = validate(model, val_loader, device)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(loader):.4f}, Val Acc: {val_acc:.4f}")

    # Guardar modelo al final
    mlflow.pytorch.log_model(model, "model")
```

Esto genera series temporales visibles en la UI como gráficos de línea. Útil para detectar overfitting (divergencia train/val) y seleccionar el mejor epoch checkpoint.

---

## 9. 📂 Organización de Experimentos

Estrategias recomendadas para equipos:

| Estrategia | Formato | Ejemplo | Cuándo usarla |
|---|---|---|---|
| **Por proyecto** | `{project_id}/{task}` | `nlp/sentiment` | Equipos con múltiples proyectos |
| **Por dataset** | `{dataset}_{version}` | `iris_v2` | Mismo modelo, datasets cambiantes |
| **Por modelo** | `{model_family}/{variant}` | `rf/baseline`, `rf/optimized` | Comparación de arquitecturas |
| **Por sprint** | `{sprint_id}/{ticket}` | `Sprint_14/ML-203` | Entornos ágiles con tracking Jira |

### Taxonomía de Runs

```
runs/
├── {project}/
│   ├── {model}_{yyyy-mm-dd}_{git_hash}/
│   │   ├── baseline/
│   │   ├── lr_tune/
│   │   └── final/
```

Convención de tags:

```python
mlflow.set_tag("git_commit", "a1b2c3d")
mlflow.set_tag("branch", "feature/experimental")
mlflow.set_tag("dataset_version", "v2.1")
mlflow.set_tag("pipeline_stage", "train")
mlflow.set_tag("trigger", "manual" | "CI" | "scheduled")
```

---

## 10. 🚀 MLflow en Producción

### Servidor de Tracking para Equipos

```dockerfile
# Dockerfile para MLflow Tracking Server
FROM python:3.11-slim

RUN pip install mlflow==2.18.0 psycopg2-binary boto3

ENV BACKEND_STORE_URI=postgresql://user:pass@postgres:5432/mlflow
ENV ARTIFACT_ROOT=s3://mlflow-artifacts-bucket

EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "${BACKEND_STORE_URI}", \
     "--default-artifact-root", "${ARTIFACT_ROOT}", \
     "--host", "0.0.0.0"]
```

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${MLFLOW_DB_PASS}

  mlflow:
    build: .
    ports: ["5000:5000"]
    depends_on: [postgres]
    environment:
      MLFLOW_DB_PASS: ${MLFLOW_DB_PASS}
      AWS_ACCESS_KEY_ID: ${AWS_KEY}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET}
```

### Despliegue en Kubernetes

```
┌─────────────────────────────────────────────────────┐
│ K8s Cluster                                         │
│  ┌──────────┐  ┌─────────────┐  ┌───────────────┐ │
│  │ MLflow   │  │ PostgreSQL  │  │  S3 (MinIO/   │ │
│  │ Pod      │  │ StatefulSet │  │  AWS/GCS)     │ │
│  │ :5000    │  │ :5432       │  │               │ │
│  └──────────┘  └─────────────┘  └───────────────┘ │
│  ┌──────────────────────────────────────────────┐  │
│  │ Ingress: mlflow.company.com                 │  │
│  │ Auth: OAuth2 Proxy / NGINX basic auth       │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

Puntos clave para producción:
- **Backend Store:** PostgreSQL (nunca SQLite para multi-usuario)
- **Artifact Store:** S3/GCS con IAM roles, nunca filesystem compartido
- **Autenticación:** NGINX reverse proxy con OAuth2-Proxy o basic auth
- **Monitoreo:** Prometheus + Grafana para métricas del servidor MLflow
- **Backups:** PostgreSQL pg_dump programado + versionado de bucket S3

---

## 11. 🌍 Aplicaciones Reales

| Empresa | Caso de Uso | Detalle |
|---|---|---|
| **Spotify** | Recomendación musical | Miles de runs diarios comparando variantes de embeddings sin duplicar infraestructura |
| **Databricks** | AutoML y Feature Store | MLflow integrado nativamente con Spark, Delta Lake y Unity Catalog |
| **Zynga** | Predicción de churn en juegos | Tracking de cientos de experimentos de feature engineering por sprint |
| **Regeneron** | Drug Discovery con ML | Reproducibilidad regulatoria: cada run es auditable (FDA compliance) |
| **CERN** | Física de partículas (LHC) | HPO massivo con MLflow + Kubernetes para optimización de detectores |
| **NVIDIA** | Computer Vision (RAPIDS) | Autologging GPU-aware para métricas de memoria y throughput |
| **Airbnb** | Pricing dinámico | Pipelines de MLflow tracking integrados con Airflow para reentrenamiento |

---

## ⚠️ Advertencias y Pitfalls

- **No logues datasets completos como artifacts** si superan los límites de tu artifact store. Para grandes volúmenes, usa DVC y registra solo la referencia (hash).
- **Autologging puede capturar parámetros sensibles** del entorno (rutas, API keys). Revisa siempre qué se registra.
- **SQLite no escala para equipos.** Dos escritores concurrentes en SQLite causan deadlocks. Migra a PostgreSQL desde el día 1 si hay más de un usuario.
- **No uses `file://` en producción.** El artifact store local no es accesible por otros pods/contenedores.
- **El tracking URI debe ser compartido por todo el equipo.** Si cada miembro usa su propio servidor/local, los experimentos son invisibles entre sí.
- **Los modelos no versionados automáticamente entran al Registry sin metadatos.** Siempre asocia un run de tracking al registro de un modelo.

---

## 💡 Tips y Mejores Prácticas

- **Vincula cada run al commit exacto:** `mlflow.set_tag("git_commit", os.getenv("GIT_COMMIT"))`.
- **Usa `mlflow.search_runs()`** programáticamente para reportes automáticos y selección de best model.
- **Habilita signature logging:** `mlflow.models.infer_signature()` ayuda a validar schemas al hacer inference.
- **Nested runs para HPO:** `mlflow.start_run(nested=True)` crea jerarquías de runs padre/hijo útiles para hyperparameter search.
- **Artifact URI canónico:** Define `MLFLOW_TRACKING_URI` como variable de entorno o en `.env` del proyecto. Así evitás hardcodear URIs en cada script.
- **Limpiar runs obsoletos:** `mlflow.delete_run(run_id)` para experimentos descartados. Mantiene limpia la UI y la base de datos.
- **Usa `mlflow.evaluate()`** para evaluaciones estandarizadas con métricas predefinidas y artefactos de evaluación.

---

## 📦 Código de Compresión

```python
# Pipeline tracked mínimo funcional
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_quickstart")

with mlflow.start_run(run_name="rf_baseline"):
    mlflow.autolog()

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    mlflow.log_metric("accuracy", accuracy_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}")
```

---

## ✅ Verificación de Conocimiento

1. **¿Qué diferencia hay entre un `log_param()` y un `set_tag()`?** — Un param es variable de configuración del modelo, un tag es metadata inmutable (commit, versión de datos). Los params son numéricos/string comparables en UI; los tags son solo strings para filtrado.

2. **¿Por qué SQLite no escala para equipos?** — SQLite usa bloqueos de escritura exclusivos. Dos usuarios creando runs simultáneamente causan timeouts. PostgreSQL permite escritura concurrente con MVCC.

3. **¿Cuál es la diferencia entre Backend Store y Artifact Store?** — El backend store guarda metadatos ligeros (params, metrics, tags) en base de datos relacional. El artifact store guarda blobs binarios (modelos, gráficos) en object storage.

4. **¿Qué ventaja tiene MLflow sobre Sacred?** — MLflow incluye UI integrada, Model Registry, autologging, y Pipelines. Sacred requiere Omniboard para UI y no tiene Model Registry nativo.

5. **¿Cuándo usar autologging vs logging manual?** — Autologging para exploración rápida y desarrollo. Logging manual para producción donde necesitas control granular sobre qué se registra y cómo.

---

## 🎯 Key Takeaways

- MLflow Tracking responde **qué**, **cómo** y **con qué resultado** se ejecutó cada experimento.
- La triada Backend Store (metadatos) + Artifact Store (blobs) + UI (visualización) forma el núcleo de tracking productivo.
- **Nunca uses SQLite ni filesystem local para equipos.** PostgreSQL + S3 desde el inicio.
- MLflow Projects empaquetan código reproducible; MLflow Pipelines estandarizan workflows completos.
- El autologging es un acelerador, no un reemplazo del logging explícito en pipelines productivos.
- El [[03 - Model Registry y Lifecycle|Model Registry]] es el bridge natural: el mejor run del tracking se promociona a Staging → Production.

---

## Referencias

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API Reference](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Pipelines](https://mlflow.org/docs/latest/pipelines.html)
- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
- Sculley et al. "Hidden Technical Debt in Machine Learning Systems" (NIPS 2015)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
