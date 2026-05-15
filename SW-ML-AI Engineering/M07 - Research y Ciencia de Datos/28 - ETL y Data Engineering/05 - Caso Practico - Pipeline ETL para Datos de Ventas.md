# 🎯 Caso Practico - Pipeline ETL para Datos de Ventas

Este proyecto integra todos los conceptos aprendidos en el módulo para construir un **pipeline ETL completo y listo para producción**. El objetivo es procesar datos de ventas provenientes de múltiples fuentes heterogéneas, transformarlos aplicando reglas de negocio y feature engineering, y cargarlos en un data warehouse dimensional. Este pipeline servirá como fuente de verdad tanto para dashboards de negocio como para alimentar modelos de Machine Learning de predicción de demanda y segmentación de clientes.

![Sales Pipeline Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Data_integration.jpg/640px-Data_integration.jpg)

## 1. Requisitos del Proyecto

### 1.1 Requisitos Funcionales

1. Extraer datos diariamente desde tres fuentes:
   - Archivos CSV generados por tiendas físicas.
   - API REST del sistema de comercio electrónico.
   - Base de datos PostgreSQL del ERP corporativo.

2. Aplicar transformaciones consistentes:
   - Estandarización de formatos de moneda y fecha.
   - Limpieza de registros inválidos o incompletos.
   - Enriquecimiento con datos de productos y clientes.
   - Cálculo de métricas derivadas (margen, descuento efectivo).

3. Cargar datos en un data warehouse dimensional:
   - Esquema estrella con tablas de hechos y dimensiones.
   - Soporte para SCD Tipo 2 en dimensiones de cliente y producto.
   - Particionamiento por fecha para optimizar consultas.

4. Orquestar y monitorear:
   - Ejecución programada cada 6 horas.
   - Alertas en caso de fallo o degradación de calidad.
   - Validación de datos antes de la carga final.

### 1.2 Requisitos No Funcionales

- **Escalabilidad:** Debe soportar un crecimiento de 10K a 10M registros diarios sin rediseño arquitectónico.
- **Idempotencia:** Re-ejecuciones del mismo período no deben generar duplicados.
- **Recuperación ante fallos:** Checkpointing y reintentos automáticos.
- **Auditabilidad:** Log completo de transformaciones y metadatos de ejecución.

## 2. Arquitectura del Sistema

```mermaid
graph TD
    subgraph Fuentes
        A1[CSV Tiendas]
        A2[API E-commerce]
        A3[PostgreSQL ERP]
    end

    subgraph Ingesta
        B1[Extracción Python]
        B2[Validación Inicial]
    end

    subgraph Transformación
        C1[Limpieza y Normalización]
        C2[Enriquecimiento]
        C3[Feature Engineering]
    end

    subgraph Carga
        D1[Staging Area]
        D2[Data Warehouse Dimensional]
    end

    subgraph Orquestación
        E1[Apache Airflow]
        E2[Great Expectations]
        E3[AlertManager]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D1
    D1 --> E2
    E2 -->|Valid OK| D2
    E2 -->|Valid FAIL| E3
    E1 --> B1
    E1 --> C1
    E1 --> D2
```

## 3. Diseño del Data Warehouse Dimensional

### 3.1 Tablas de Dimensiones

**Dim_Cliente (SCD Tipo 2):**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| cliente_sk | INT | Surrogate Key |
| cliente_id_natural | VARCHAR | ID del ERP |
| nombre | VARCHAR | Nombre completo |
| email | VARCHAR | Correo electrónico |
| segmento | VARCHAR | Premium, Standard, Basic |
| ciudad | VARCHAR | Ciudad de residencia |
| fecha_inicio | DATE | Inicio de validez |
| fecha_fin | DATE | Fin de validez |
| activo | BOOLEAN | Registro actual |

**Dim_Producto (SCD Tipo 2):**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| producto_sk | INT | Surrogate Key |
| producto_id_natural | VARCHAR | SKU |
| nombre | VARCHAR | Nombre del producto |
| categoria | VARCHAR | Categoría jerárquica |
| subcategoria | VARCHAR | Subcategoría |
| marca | VARCHAR | Marca |
| costo_base | DECIMAL | Costo de fabricación |
| precio_base | DECIMAL | Precio de venta sugerido |
| fecha_inicio | DATE | Inicio de validez |
| fecha_fin | DATE | Fin de validez |
| activo | BOOLEAN | Registro actual |

**Dim_Tiempo:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fecha_sk | INT | Surrogate Key (YYYYMMDD) |
| fecha | DATE | Fecha calendario |
| anio | INT | Año |
| trimestre | INT | Q1, Q2, Q3, Q4 |
| mes | INT | Mes (1-12) |
| dia | INT | Día del mes |
| dia_semana | INT | 0=Lunes, 6=Domingo |
| es_fin_de_semana | BOOLEAN | Sábado o domingo |
| es_feriado | BOOLEAN | Festivo nacional |

### 3.2 Tabla de Hechos

**Fact_Ventas:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| venta_sk | BIGINT | Surrogate Key auto-incremental |
| fecha_sk | INT | FK a Dim_Tiempo |
| cliente_sk | INT | FK a Dim_Cliente |
| producto_sk | INT | FK a Dim_Producto |
| canal_sk | INT | FK a Dim_Canal |
| ubicacion_sk | INT | FK a Dim_Ubicacion |
| cantidad | INT | Unidades vendidas |
| precio_unitario | DECIMAL | Precio real de venta |
| costo_unitario | DECIMAL | Costo del producto |
| descuento | DECIMAL | Porcentaje de descuento aplicado |
| ingreso_neto | DECIMAL | cantidad * precio_unitario * (1 - descuento) |
| margen_bruto | DECIMAL | ingreso_neto - (cantidad * costo_unitario) |
| metodo_pago | VARCHAR | Tarjeta, Efectivo, Transferencia |

## 4. Implementación del Pipeline

### 4.1 Extracción Multi-Fuente

```python
"""
Fase de Extracción: Lectura desde CSV, API y PostgreSQL.
"""

import pandas as pd
import requests
from sqlalchemy import create_engine
from datetime import datetime, timedelta

class Extractor:
    def __init__(self, execution_date):
        self.execution_date = execution_date
        self.raw_data = {}

    def extract_csv(self, path):
        """Extrae datos de ventas de tiendas físicas."""
        df = pd.read_csv(path, parse_dates=['transaction_date'])
        # Filtrar solo el período de ejecución
        df = df[df['transaction_date'].dt.date == self.execution_date]
        self.raw_data['csv_sales'] = df
        print(f"CSV: Extraídos {len(df)} registros.")
        return self

    def extract_api(self, base_url, api_key):
        """Extrae datos del e-commerce vía REST API."""
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "date_from": self.execution_date.isoformat(),
            "date_to": (self.execution_date + timedelta(days=1)).isoformat(),
            "limit": 10000
        }
        response = requests.get(f"{base_url}/orders", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()['orders']
        df = pd.DataFrame(data)
        self.raw_data['api_orders'] = df
        print(f"API: Extraídos {len(df)} registros.")
        return self

    def extract_postgres(self, connection_string, query):
        """Extrae datos del ERP."""
        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine, params={"exec_date": self.execution_date})
        self.raw_data['erp_data'] = df
        print(f"PostgreSQL: Extraídos {len(df)} registros.")
        return self

# Uso:
# extractor = Extractor(execution_date=datetime(2024, 5, 1).date())
# extractor.extract_csv("data/sales_stores.csv") \
#           .extract_api("https://api.ecommerce.com", "secret_key") \
#           .extract_postgres("postgresql://user:pass@host/db", "SELECT * FROM sales WHERE date = %(exec_date)s")
```

### 4.2 Transformación y Feature Engineering

```python
"""
Fase de Transformación: Limpieza, normalización y enriquecimiento.
"""

import numpy as np

class Transformer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.transformed = None

    def normalize_csv(self, df):
        """Estandariza columnas del CSV."""
        df = df.rename(columns={
            'transaction_date': 'fecha',
            'store_id': 'ubicacion_id',
            'product_code': 'producto_id',
            'customer_code': 'cliente_id'
        })
        df['canal'] = 'fisico'
        return df

    def normalize_api(self, df):
        """Estandariza columnas de la API."""
        df = df.rename(columns={
            'order_date': 'fecha',
            'sku': 'producto_id',
            'user_id': 'cliente_id',
            'quantity': 'cantidad',
            'unit_price': 'precio_unitario'
        })
        df['canal'] = 'online'
        df['ubicacion_id'] = 'web'
        return df

    def normalize_erp(self, df):
        """Estandariza columnas del ERP."""
        df = df.rename(columns={
            'sale_timestamp': 'fecha',
            'product_sku': 'producto_id',
            'customer_uuid': 'cliente_id'
        })
        df['canal'] = df['channel_type']
        return df

    def merge_sources(self):
        """Une todas las fuentes en un solo dataframe."""
        frames = []
        if 'csv_sales' in self.raw_data:
            frames.append(self.normalize_csv(self.raw_data['csv_sales']))
        if 'api_orders' in self.raw_data:
            frames.append(self.normalize_api(self.raw_data['api_orders']))
        if 'erp_data' in self.raw_data:
            frames.append(self.normalize_erp(self.raw_data['erp_data']))
        merged = pd.concat(frames, ignore_index=True)
        return merged

    def clean_and_enrich(self, df, products_df, customers_df):
        """Limpieza y enriquecimiento."""
        # Eliminar registros sin cantidad o precio
        df = df.dropna(subset=['cantidad', 'precio_unitario'])
        df = df[df['cantidad'] > 0]
        df = df[df['precio_unitario'] > 0]

        # Normalizar fechas
        df['fecha'] = pd.to_datetime(df['fecha']).dt.date

        # Join con dimensión producto para obtener costo
        df = df.merge(products_df[['producto_id', 'costo_base', 'categoria']],
                      on='producto_id', how='left')

        # Join con dimensión cliente para segmento
        df = df.merge(customers_df[['cliente_id', 'segmento']],
                      on='cliente_id', how='left')

        # Feature: descuento implícito si hay precio base de referencia
        # (Simulado: descuento = 1 - precio_unitario / (costo_base * 1.3))
        df['descuento'] = np.clip(1 - (df['precio_unitario'] / (df['costo_base'] * 1.3)), 0, 0.5)
        df['descuento'] = df['descuento'].fillna(0)

        # Métricas derivadas
        df['ingreso_neto'] = df['cantidad'] * df['precio_unitario'] * (1 - df['descuento'])
        df['margen_bruto'] = df['ingreso_neto'] - (df['cantidad'] * df['costo_base'])

        # Feature para ML: ratio de margen
        df['margen_ratio'] = df['margen_bruto'] / df['ingreso_neto']
        df['margen_ratio'] = df['margen_ratio'].fillna(0)

        self.transformed = df
        print(f"Transformados {len(df)} registros válidos.")
        return self

# Uso:
# transformer = Transformer(extractor.raw_data)
# merged = transformer.merge_sources()
# transformer.clean_and_enrich(merged, products_df, customers_df)
```

### 4.3 Carga en el Data Warehouse

```python
"""
Fase de Carga: Inserción incremental en el esquema estrella.
"""

from sqlalchemy import Table, MetaData, select

class Loader:
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()

    def load_dimension(self, df, table_name, natural_key, surrogate_key):
        """
        Carga incremental con soporte para SCD Tipo 2.
        Detecta cambios en el registro activo y cierra la versión anterior.
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)

        with self.engine.begin() as conn:
            for _, row in df.iterrows():
                # Buscar registro activo actual
                stmt = select(table).where(
                    (table.c[natural_key] == row[natural_key]) &
                    (table.c['activo'] == True)
                )
                existing = conn.execute(stmt).fetchone()

                if existing is None:
                    # Nuevo registro
                    row_dict = row.to_dict()
                    row_dict['fecha_inicio'] = datetime.now().date()
                    row_dict['fecha_fin'] = datetime(9999, 12, 31).date()
                    row_dict['activo'] = True
                    conn.execute(table.insert(), row_dict)
                elif self.has_changes(existing, row):
                    # Cerrar registro anterior
                    conn.execute(
                        table.update().where(table.c[surrogate_key] == existing[surrogate_key]),
                        {'fecha_fin': datetime.now().date(), 'activo': False}
                    )
                    # Insertar nuevo registro
                    row_dict = row.to_dict()
                    row_dict['fecha_inicio'] = datetime.now().date()
                    row_dict['fecha_fin'] = datetime(9999, 12, 31).date()
                    row_dict['activo'] = True
                    conn.execute(table.insert(), row_dict)
        print(f"Dimensión {table_name} actualizada.")

    def has_changes(self, existing, new_row):
        """Compara atributos relevantes para detectar cambios."""
        for col in ['nombre', 'segmento', 'ciudad', 'categoria', 'marca']:
            if col in new_row and existing[col] != new_row[col]:
                return True
        return False

    def load_fact(self, df, table_name):
        """Carga incremental en tabla de hechos."""
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        # Asumiendo que tenemos los surrogate keys resueltos previamente
        df.to_sql(table_name, self.engine, if_exists='append', index=False)
        print(f"Cargados {len(df)} registros en {table_name}.")

# Uso:
# loader = Loader(engine)
# loader.load_dimension(customers_dim, 'dim_cliente', 'cliente_id', 'cliente_sk')
# loader.load_fact(transformed_facts, 'fact_ventas')
```

## 5. Orquestación con Apache Airflow

```python
"""
DAG de Airflow para orquestar el pipeline ETL de ventas.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

def run_extraction(**context):
    execution_date = context['ds']
    print(f"Ejecutando extracción para {execution_date}")
    # Lógica de extracción

def run_transformation(**context):
    print("Ejecutando transformación y feature engineering")
    # Lógica de transformación

def run_validation(**context):
    print("Ejecutando validación de calidad con Great Expectations")
    # Llamada a GE

def run_load(**context):
    print("Ejecutando carga en data warehouse")
    # Lógica de carga

def notify_failure(context):
    """Callback para alertas."""
    print(f"🚨 Pipeline falló en la tarea: {context['task_instance'].task_id}")
    # Enviar alerta a Slack/PagerDuty

with DAG(
    dag_id='pipeline_etl_ventas_ml',
    default_args={
        'owner': 'data_engineering',
        'depends_on_past': True,
        'email_on_failure': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=10),
        'on_failure_callback': notify_failure,
    },
    description='Pipeline ETL completo para datos de ventas multi-fuente.',
    schedule_interval='0 */6 * * *',  # Cada 6 horas
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'ventas', 'ml', 'produccion'],
) as dag:

    extract = PythonOperator(
        task_id='extraccion_fuentes',
        python_callable=run_extraction,
    )

    transform = PythonOperator(
        task_id='transformacion_enriquecida',
        python_callable=run_transformation,
    )

    validate = PythonOperator(
        task_id='validacion_calidad',
        python_callable=run_validation,
    )

    load = PythonOperator(
        task_id='carga_warehouse',
        python_callable=run_load,
    )

    extract >> transform >> validate >> load
```

## 6. Métricas del Proyecto

| Métrica | Objetivo | Instrumento |
|---------|----------|-------------|
| Latencia end-to-end | < 30 minutos | Airflow UI |
| Tasa de registros rechazados | < 0.1% | Great Expectations |
| Disponibilidad del pipeline | 99.9% | Uptime monitor |
| Tiempo de recuperación (RTO) | < 15 minutos | Alertas + Runbooks |
| Freshness de datos | < 6 horas | SLA en Airflow |

> 💡 **Tip:** Durante el desarrollo de este proyecto, utiliza **dbt (data build tool)** para gestionar las transformaciones SQL en el warehouse. dbt permite versionar tu lógica de transformación, documentar dependencias y ejecutar tests automáticos directamente en tu data warehouse, convirtiendo el SQL en código software de primera clase.

## 7. Código de Compresión

```python
"""
📦 Pipeline ETL Ventas - Versión Compacta Integrada
Orquesta extracción, transformación y carga en un solo flujo.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Configuración
DB_URI = "postgresql://user:pass@localhost:5432/warehouse"
engine = create_engine(DB_URI)

def run_pipeline(csv_path, products_df, customers_df, exec_date):
    # 1. EXTRACT
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[df['date'].dt.date == exec_date]

    # 2. TRANSFORM
    df = df.dropna(subset=['quantity', 'price'])
    df = df[(df['quantity'] > 0) & (df['price'] > 0)]
    df['revenue'] = df['quantity'] * df['price']
    df = df.merge(products_df[['sku', 'cost']], left_on='product_id', right_on='sku', how='left')
    df = df.merge(customers_df[['id', 'segment']], left_on='customer_id', right_on='id', how='left')
    df['margin'] = df['revenue'] - (df['quantity'] * df['cost'])
    df['margin_ratio'] = df['margin'] / df['revenue']

    # 3. LOAD (Append idempotente usando INSERT ... ON CONFLICT)
    df[['date', 'customer_id', 'product_id', 'quantity', 'price', 'revenue', 'margin', 'margin_ratio']].to_sql(
        'staging_sales', engine, if_exists='append', index=False
    )
    print(f"Pipeline completado: {len(df)} registros procesados.")
    return len(df)

# Uso:
# run_pipeline("data/sales.csv", products_df, customers_df, datetime(2024, 5, 1).date())
```

## 8. 🎯 Proyecto Documentado

### 8.1 Estructura del Repositorio

```
etl_ventas_ml/
├── dags/
│   └── dag_pipeline_ventas.py      # Orquestación Airflow
├── src/
│   ├── extract.py                  # Lógica de extracción
│   ├── transform.py                # Lógica de transformación
│   ├── load.py                     # Lógica de carga
│   └── quality.py                  # Validación con GE
├── models/
│   └── dbt/                        # Transformaciones SQL con dbt
├── config/
│   └── connections.yaml            # Credenciales y URIs
├── tests/
│   └── test_transformations.py     # Unit tests
├── docs/
│   ├── arquitectura.md             # Diagramas y decisiones
│   └── runbook.md                  # Procedimientos operativos
└── requirements.txt
```

### 8.2 Decisiones Arquitectónicas Clave (ADRs)

1. **ADR-001: Uso de Esquema Estrella sobre Copo de Nieve**
   - **Contexto:** Necesitamos simplificar los joins para los científicos de datos que extraen datasets de entrenamiento.
   - **Decisión:** Esquema estrella con dimensiones desnormalizadas.
   - **Consecuencias:** Mayor redundancia pero queries 40% más rápidas.

2. **ADR-002: SCD Tipo 2 para Clientes y Productos**
   - **Contexto:** Los modelos de ML requieren reconstruir el estado del cliente en cualquier fecha histórica.
   - **Decisión:** Implementar SCD Tipo 2 con rangos de fecha.
   - **Consecuencias:** Complejidad en la carga, pero capacidad de time-travel para features temporales.

3. **ADR-003: Particionamiento por Fecha en Fact_Ventas**
   - **Contexto:** Las consultas de entrenamiento filtran típicamente por rangos de fecha.
   - **Decisión:** Particionar la tabla de hechos por año y mes.
   - **Consecuencias:** Reduce el costo de scan en un 85% para queries mensuales.

4. **ADR-004: ELT sobre ETL para Transformaciones SQL**
   - **Contexto:** El equipo de analytics prefiere SQL sobre Python para lógica de negocio.
   - **Decisión:** Cargar datos en staging y transformar con dbt/SQL dentro del warehouse.
   - **Consecuencias:** Menor costo de compute en Spark, mayor aprovechamiento del warehouse.

### 8.3 Próximos Pasos y Evolución

1. **Migración a Lakehouse:** Evaluar Delta Lake sobre S3 para unificar batch y streaming.
2. **Feature Store:** Integrar Feast o Tecton para servir las features calculadas directamente a los modelos de ML.
3. **Data Contracts:** Implementar contratos de datos entre productores y consumidores usando herramientas como Schemata o Protobuf.
4. **Monitoreo de Drift:** Añadir validación estadística (KS-test, PSI) en el pipeline para detectar drift en las distribuciones de features antes de que afecte el modelo.

---

Este caso práctico demuestra cómo los pilares técnicos de ETL, procesamiento distribuido, modelado dimensional y gobernanza se fusionan para crear un sistema de datos robusto. Domina estos conceptos y estarás preparado para diseñar arquitecturas de datos que escalan con las demandas de la IA moderna.
