# 🐼 Pandas para Análisis de Datos

## 🎯 Learning Objectives

- Cargar un CSV con `pd.read_csv` y diagnosticar su estructura con `head`, `info`, `describe` en menos de 5 líneas
- Seleccionar columnas (`df['col']`, `df[['c1','c2']]`) y filtrar filas con **boolean indexing** (`df[df['age']>30]`, `&`/`|`/`~`)
- Dominar **`df.groupby().agg()`** para responder "¿media de X por categoría?" — la pregunta más frecuente en pruebas técnicas
- Manejar datos faltantes con `isnull`, `fillna`, `dropna` y entender cuándo cada uno es apropiado
- Combinar dos DataFrames con `merge` (JOIN SQL) y `concat` (apilar filas/columnas)
- Aplicar funciones custom con `df.apply` y `df['col'].map` para transformaciones vectorizadas
- Conocer `value_counts`, `sort_values`, `unique` para análisis exploratorio rápido

---

## Introduction

Si NumPy es la capa de **cómputo numérico**, Pandas es la capa de **datos tabulares**. En cualquier prueba técnica de IA, después de "crea un array y suma por columnas" viene "carga este CSV, filtra los mayores de 30, agrupa por categoría y dime la media". Pandas es la herramienta para esa segunda parte, y aunque su API es más grande que la de NumPy, el 80% del trabajo se hace con 10-15 métodos que se aprenden en una tarde.

La nota sigue el flujo natural de un análisis de datos: **cargar → inspeccionar → limpiar → transformar → agrupar → exportar**. Cada método viene con su caso real y un antipatrón común que la gente escribe los primeros meses.

Conectamos con [[08 - NumPy para Analisis de Datos]] (Pandas se construye sobre NumPy: cada columna es un `np.ndarray`), con [[05 - Json y Pickle]] (la forma más común de llevar datos JSON a un DataFrame es `pd.read_json` o `pd.DataFrame.from_dict`), y con [[05 - Manejo de Archivos]] (el `csv` stdlib es lo que Pandas automatiza bajo el capó).

---

## 1. The Problem and Why This Solution Exists

### El problema de los datos tabulares

Tienes un CSV con 50,000 filas y 12 columnas. Quieres responder "¿cuál es la edad promedio de los sobrevivientes del Titanic por clase?". Con listas de Python:

```python
# Patrón naive con listas de diccionarios
import csv
with open("titanic.csv") as f:
    filas = list(csv.DictReader(f))

sobrevivientes_primera = [float(f["Age"]) for f in filas
                          if f["Survived"] == "1" and f["Pclass"] == "1"]
media = sum(sobrevivientes_primera) / len(sobrevivientes_primera)
```

Funciona, pero tienes que escribir el filtro, el manejo de nulos y la agregación manualmente. Con DataFrames:

```python
import pandas as pd
df = pd.read_csv("titanic.csv")
media = df[(df["Survived"] == 1) & (df["Pclass"] == 1)]["Age"].mean()
```

Misma respuesta, **una línea, con manejo automático de NaN**. Ese es el punto de Pandas.

### El modelo mental: Series y DataFrame

- **Series**: array 1D etiquetado (como un `dict` con orden). Es lo que obtienes al hacer `df['col']`.
- **DataFrame**: tabla 2D etiquetada. Cada columna es una Series. Las filas y columnas tienen **índices** que permiten acceso posicional (`.iloc`) o por etiqueta (`.loc`).

Piensa en un DataFrame como **un dict de Series que comparten el mismo índice de filas**. Esa metáfora explica por qué `df['col']` devuelve una columna y por qué la alineación de índices hace que las operaciones vectorizadas "simplemente funcionen".

---

## 2. Conceptual Deep Dive

### 2.1 Cargar y diagnosticar: las 5 primeras líneas

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

df.head()        # primeras 5 filas (df.head(10) para 10)
df.tail(3)       # últimas 3
df.shape         # (891, 12)  ← tupla (filas, columnas)
df.columns       # Index(['PassengerId', 'Survived', ...], dtype='object')
df.dtypes        # PassengerId      int64
                 # Survived         int64
                 # Name            object
                 # Age            float64
                 # ...
df.info()        # resumen: tipos + count de no-nulos + memoria
df.describe()    # count, mean, std, min, 25%, 50%, 75%, max para columnas numéricas
df.describe(include="all")  # incluye categóricas (count, unique, top, freq)
```

💡 **Tip mental**: si la prueba te da un CSV y no dice nada, **`df.head()` + `df.info()`** es siempre la primera respuesta. Te dice el nombre de las columnas, sus tipos y cuántos nulos hay — con eso resuelves el 50% de las preguntas.

### 2.2 Selección de columnas y filtrado de filas

```python
# Selección de UNA columna → Series
df["Age"]                          # Series (1D)
df.Age                             # mismo, pero no usar si el nombre tiene espacios

# Selección de VARIAS columnas → DataFrame
df[["Name", "Age", "Sex"]]         # ojo al doble corchete

# Filtrado booleano: filas donde se cumple una condición
df[df["Age"] > 30]                 # sobrevivientes con Age > 30

# Múltiples condiciones: OBLIGATORIO usar paréntesis y & / | / ~
adultos_primera = df[(df["Age"] >= 18) & (df["Pclass"] == 1)]
ninos_o_mujeres = df[(df["Age"] < 18) | (df["Sex"] == "female")]
no_tercera = df[~(df["Pclass"] == 3)]

# .loc (etiqueta) y .iloc (posición)
df.loc[0:5, ["Name", "Age"]]       # filas 0-5, columnas Name y Age (POR ETIQUETA)
df.iloc[0:5, [2, 4]]               # filas 0-5, columnas 2 y 4 (POR POSICIÓN)
```

⚠️ **Advertencia — `&` no `and`**: en boolean indexing de Pandas, los operadores son `&`, `|` y `~` (element-wise). Usar `and`, `or`, `not` (booleanos de Python) produce `ValueError: The truth value of a Series is ambiguous`.

### 2.3 El método estrella: `groupby().agg()`

```python
# "¿Media de edad por clase?"
df.groupby("Pclass")["Age"].mean()
# Pclass
# 1    38.23
# 2    29.87
# 3    25.14

# Múltiples agregaciones
df.groupby("Pclass")["Age"].agg(["mean", "median", "count"])

# Múltiples columnas, múltiples funciones
df.groupby("Pclass").agg(
    edad_media=("Age", "mean"),
    tarifa_media=("Fare", "median"),
    sobrevivientes=("Survived", "sum"),
)

# Groupby por múltiples columnas
df.groupby(["Pclass", "Sex"])["Survived"].mean()
# Pclass  Sex
# 1       female    0.96
#         male      0.36
# 2       female    0.92
#         male      0.15
# 3       female    0.50
#         male      0.13
```

La sintaxis moderna (Pandas 0.25+) usa **named aggregation** dentro de `.agg()`: `nombre_nueva_col=("col_original", "funcion")`. Es más legible que un dict.

### 2.4 Datos faltantes: `isnull`, `fillna`, `dropna`

```python
# Detectar nulos
df.isnull()                # DataFrame de True/False
df.isnull().sum()          # cuenta de nulos por columna
df["Age"].isnull()         # Series booleana: True donde Age es NaN

# Eliminar filas con nulos
df.dropna()                # cualquier NaN en cualquier columna
df.dropna(subset=["Age"])  # solo si Age es NaN
df.dropna(thresh=10)       # mantener filas con al menos 10 valores no-nulos

# Rellenar nulos
df["Age"].fillna(0)                    # con un escalar
df["Age"].fillna(df["Age"].median())   # con la mediana (estrategia común)
df["Age"].fillna(method="ffill")       # forward-fill: propaga el valor anterior
df["Age"].fillna(method="bfill")       # backward-fill
```

💡 **Tip**: en pruebas técnicas, "llena los nulos con la mediana de la columna" es la respuesta correcta el 80% de las veces. La media es sensible a outliers; la mediana es robusta.

### 2.5 Combinar DataFrames: `merge` y `concat`

```python
# merge: equivalente a SQL JOIN
usuarios = pd.DataFrame({"id": [1, 2, 3], "nombre": ["Ana", "Bob", "Clara"]})
compras = pd.DataFrame({"user_id": [1, 1, 2], "monto": [100, 50, 200]})

# INNER JOIN (default)
usuarios.merge(compras, left_on="id", right_on="user_id")

# LEFT JOIN
usuarios.merge(compras, left_on="id", right_on="user_id", how="left")

# concat: apilar filas o columnas
pd.concat([df_2023, df_2024])              # apilar filas (axis=0, default)
pd.concat([df_features, df_target], axis=1)  # apilar columnas (axis=1)
```

| `how` | Significado |
|---|---|
| `"inner"` | Solo filas con match en ambos (default) |
| `"left"` | Todas las filas del izquierdo + match del derecho (NaN si no) |
| `"right"` | Inverso de left |
| `"outer"` | Unión completa (NaN donde no hay match) |

### 2.6 Análisis exploratorio: el toolkit de los 30 segundos

```python
# value_counts: distribución de categorías
df["Pclass"].value_counts()           # 3    491
                                       # 1    216
                                       # 2    184
df["Pclass"].value_counts(normalize=True)  # proporciones, no counts

# unique y nunique
df["Embarked"].unique()               # array(['S', 'C', 'Q', nan])
df["Embarked"].nunique()              # 3 (no cuenta NaN)

# sort_values
df.sort_values("Age", ascending=False).head(10)         # top 10 más viejos
df.sort_values(["Pclass", "Age"]).head(20)              # multi-columna

# apply: función custom sobre cada valor (lento, usar con cuidado)
df["Name_length"] = df["Name"].apply(len)
df["Title"] = df["Name"].apply(lambda s: s.split(",")[1].split(".")[0].strip())

# map: mapeo de un dict
sexo_map = {"male": "M", "female": "F"}
df["Sex_short"] = df["Sex"].map(sexo_map)

# query: filtro legible con sintaxis tipo SQL
df.query("Age > 30 and Pclass == 1")
```

⚠️ **Advertencia — `apply` es lento**: `apply` itera por cada fila/elemento en Python puro. Para operaciones vectorizadas, prefiere `np.where`, `np.select` o métodos nativos de la Series (`.str`, `.dt`). Usa `apply` solo cuando la lógica es genuinamente no-vectorizable.

### 2.7 Exportar resultados

```python
# A CSV
df.to_csv("resultado.csv", index=False)              # index=False: no guardar el índice
df.to_csv("resultado.csv", index=False, sep=";")     # separador europeo

# A JSON
df.to_json("resultado.json", orient="records")       # lista de dicts (más portable)

# A NumPy (si necesitas entrenar un modelo)
X = df[["Age", "Fare", "Pclass"]].to_numpy()         # shape (n, 3)
y = df["Survived"].to_numpy()                        # shape (n,)

# A dict de Python
df.head(3).to_dict(orient="records")
# [{'Age': 22.0, 'Name': 'Braund, Mr. Owen Harris', ...}, ...]
```

---

## 3. Production Reality

### Memoria: por qué `pd.read_csv` puede tumbar tu laptop

Un DataFrame de 10M filas × 50 columnas en `float64` ocupa **~4 GB** solo en datos (sin overhead). Cargar el dataset completo con `pd.read_csv` es la causa #1 de `MemoryError` en pipelines de ML. Tres mitigaciones:

```python
# 1. Cargar solo las columnas que necesitas
df = pd.read_csv("huge.csv", usecols=["id", "feature_a", "feature_b"])

# 2. Especificar dtypes más baratos
df = pd.read_csv("huge.csv", dtype={"category_col": "category", "id": "int32"})

# 3. Leer por chunks (procesar y descartar)
for chunk in pd.read_csv("huge.csv", chunksize=100_000):
    process(chunk)
```

### El antipatrón del iterrows

```python
# ❌ LENTO: 100K filas = ~30 segundos
for idx, row in df.iterrows():
    if row["age"] > 30:
        df.at[idx, "high_age"] = 1

# ✅ RÁPIDO: 100K filas = ~10 ms
df["high_age"] = (df["age"] > 30).astype(int)
```

La regla: si tu bucle `for` sobre un DataFrame cabe en una operación vectorizada de Pandas, **escríbelo vectorizado**. La mejora típica es 100-1000×.

### Comparativa rápida: cuándo usar Pandas

| Tarea | Mejor herramienta | Por qué |
|---|---|---|
| CSV < 1 GB, en memoria | `pd.read_csv` | Estándar de la industria |
| CSV > 10 GB | `dask.dataframe` o Polars | Out-of-core, paralelo |
| Series temporal con 100M puntos | `polars` o `xarray` | Más rápido que Pandas para datos numéricos densos |
| Análisis ad-hoc en Jupyter | `pandas` + `pandasgui` | Mejor DX |
| Pipeline producción (ETL) | `dask` o `polars` | Más rápido, schema enforcement |

---

## 4. Code in Practice

```python
# 📦 Pandas para Análisis de Datos — Cubre: read_csv, head/info/describe,
# selección, filtrado, groupby.agg, fillna, merge, value_counts, sort_values.

import pandas as pd
import numpy as np

# 1. Cargar un dataset sintético estilo Titanic
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "PassengerId": range(1, n + 1),
    "Pclass":      np.random.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.50]),
    "Sex":         np.random.choice(["male", "female"], size=n),
    "Age":         np.random.normal(30, 14, size=n).round(1),
    "Fare":        np.random.exponential(30, size=n).round(2),
    "Embarked":    np.random.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1]),
    "Survived":    np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
})
# Inyectar nulos en Age (como en el Titanic real)
df.loc[np.random.choice(n, 20, replace=False), "Age"] = np.nan
df.loc[np.random.choice(n, 5,  replace=False), "Embarked"] = np.nan

# 2. Diagnóstico rápido (siempre lo primero)
print(f"Shape: {df.shape}")
print(f"Tipos:\n{df.dtypes}")
print(f"Nulos por columna:\n{df.isnull().sum()}")
print(f"\nDescribe (numéricas):\n{df.describe().round(2)}")

# 3. Selección + filtrado booleano
mujeres_primeraclase = df[(df["Sex"] == "female") & (df["Pclass"] == 1)]
ninas = df[df["Age"] < 18]                                # NaN en Age NO se incluye (¡importante!)

# 4. Groupby + named aggregation
resumen = df.groupby("Pclass").agg(
    n_pasajeros=("PassengerId", "count"),
    edad_media=("Age", "mean"),
    edad_mediana=("Age", "median"),
    tarifa_mediana=("Fare", "median"),
    tasa_supervivencia=("Survived", "mean"),
).round(3)
print(resumen)

# 5. Groupby multi-columna: tasa de supervivencia por clase y sexo
supervivencia = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
print(supervivencia)

# 6. Manejo de nulos: completar Age con la mediana por clase
df["Age_filled"] = df.groupby("Pclass")["Age"].transform(lambda s: s.fillna(s.median()))
print(f"Nulos antes: {df['Age'].isnull().sum()}, después: {df['Age_filled'].isnull().sum()}")

# 7. value_counts y sort_values
print("\nDistribución por puerto de embarque:")
print(df["Embarked"].value_counts(dropna=False))   # dropna=False incluye los NaN
print("\nTop 5 pasajeros más viejos:")
print(df.sort_values("Age_filled", ascending=False)[["PassengerId", "Name" if "Name" in df.columns else "Age", "Age_filled"]].head())

# 8. merge: cruzar con una tabla de tarifas promedio por clase
tarifa_promedio = df.groupby("Pclass")["Fare"].mean().rename("tarifa_promedio_clase")
df_enriquecido = df.merge(tarifa_promedio, on="Pclass")
df_enriquecido["fare_vs_promedio"] = df_enriquecido["Fare"] / df_enriquecido["tarifa_promedio_clase"]
print(df_enriquecido[["Pclass", "Fare", "tarifa_promedio_clase", "fare_vs_promedio"]].head(3))

# 9. Extraer a NumPy para alimentar un modelo
X = df_enriquecido[["Age_filled", "Fare", "Pclass"]].to_numpy()
y = df_enriquecido["Survived"].to_numpy()
print(f"\nX.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")
```

> **Caso real**: en un pipeline de feature engineering para el **Automated LLM Evaluation Suite**, los resultados de cada evaluación se almacenan como un DataFrame con columnas `[model_id, prompt_id, score, latency_ms, timestamp]`. La pregunta frecuente es "¿score medio por modelo en la última semana?" — la respuesta es una línea: `df[df['timestamp'] > one_week_ago].groupby('model_id')['score'].mean()`. La versión con listas de Python tardaría 30× más y ocuparía más memoria.

---

## 🎯 Key Takeaways

- **`pd.read_csv` + `head/info/describe`** es la secuencia de diagnóstico de los 5 segundos. Hazla primero, siempre.
- **Boolean indexing** usa `&`/`|`/`~` (NO `and`/`or`/`not`). Múltiples condiciones **siempre con paréntesis**.
- **`df.groupby().agg(nombre=("col", "funcion"))`** es la forma moderna de responder "¿X por categoría?" — más legible que dicts.
- **`fillna(df['col'].median())` es la regla del 80%** para datos faltantes numéricos. Robusto a outliers.
- **`merge` con `how="left/right/inner/outer"`** = SQL JOIN. `concat` apila filas o columnas.
- **Evita `iterrows` y `apply` cuando existe una operación vectorizada** — la diferencia es 100-1000× en velocidad.
- **`df.to_numpy()`** es el puente a NumPy / scikit-learn / PyTorch — úsalo en el límite entre "datos tabulares" y "modelo".

## References

- Pandas Getting Started: https://pandas.pydata.org/docs/getting_started/index.html
- Pandas User Guide: https://pandas.pydata.org/docs/user_guide/index.html
- Pandas Cookbook: https://pandas.pydata.org/docs/user_guide/cookbook.html
- 10 Minutes to Pandas: https://pandas.pydata.org/docs/user_guide/10min.html
- Related Vault: [[08 - NumPy para Analisis de Datos]] (Pandas se construye sobre NumPy)
- Related Vault: [[05 - Json y Pickle]] (JSON → DataFrame)
- Related Vault: [[05 - Manejo de Archivos]] (`csv` stdlib, lo que Pandas automatiza)
- Related Vault: [[../../../projects/01 - Kaggle Competitions - Project Guide|Kaggle Competitions]] (Pandas en competencias)
- Related Vault: [[../../00 - Python Avanzado para ML/07 - Caso Practico - Pipeline de Datos en Streaming|Pipeline de Datos en Streaming]] (Pandas en producción)
