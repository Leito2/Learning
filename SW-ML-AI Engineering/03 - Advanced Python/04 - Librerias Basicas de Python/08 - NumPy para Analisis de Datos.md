# 🔢 NumPy para Análisis de Datos

## 🎯 Learning Objectives

- Explicar la diferencia de **layout de memoria** entre una lista de Python y un `np.ndarray` y por qué eso implica 50-100× de velocidad
- Dominar `np.array`, `shape`, `ndim`, `dtype` y las reglas de **broadcasting** para operar arrays de formas compatibles sin bucles explícitos
- Aplicar el parámetro `axis` correctamente en `np.sum`, `np.mean`, `np.max` (el error más común en pruebas técnicas de IA)
- Generar datos sintéticos reproducibles con `np.random.seed`, `np.random.randn`, `np.random.randint`, `np.random.choice`
- Reconocer cuándo `np.where`, `np.argwhere` y el indexing booleano reemplazan un `for` sin perder claridad
- Comparar rendimiento lista vs array en una operación vectorizada y explicar la salida del profiler

---

## Introduction

Cualquier pipeline de Machine Learning — desde un CSV de Titanic hasta un batch de 70B parámetros para Llama 3 — pasa por NumPy en algún punto. En pruebas técnicas de ingreso a programas de IA (Anyone AI, entre otros), NumPy aparece como "filtro de datos": ¿sabes crear un array, entender su `shape`, sumar por filas o columnas, generar números aleatorios reproducibles? Si sí, pasas. Si no, no importa cuántos transformers sepas — el filtro de programación básica te detiene.

Esta nota cubre el mínimo que necesitas para esa prueba y para tu trabajo diario. La profundidad teórica es deliberadamente inferior a la de un curso de ML: aquí el objetivo es **destreza operativa**, no maestría. Si ya dominas las colecciones built-in de Python (listas, tuplas, dicts, sets), NumPy te resultará natural — es la misma idea (colección indexada de elementos) con tres superpoderes: **layout de memoria contiguo**, **operaciones vectorizadas** y **broadcasting**.

Conectamos con [[01 - Math y Random]] (cuándo basta con `math`/`random` y cuándo necesitas NumPy), con [[05 - Computer Vision Pipeline]] (donde NumPy se usa intensivamente para mAP, IoU, cumulativos) y con [[../05 - Librerias Especificas]] (Requests, Sqlite3, Pathlib — NumPy no es stdlib pero vive en el mismo orbit de "herramientas que tocas a diario").

---

## 1. The Problem and Why This Solution Exists

### Por qué una lista de Python es lenta

Cuando escribes `lista = [1, 2, 3, 4]` en Python puro, lo que realmente pasa en memoria es:

1. Python crea **cuatro objetos `int`** separados en el heap (uno por cada número).
2. Python crea **un objeto `list`** que contiene **cuatro punteros de 8 bytes** apuntando a esos `int`.
3. Cualquier operación aritmética debe seguir el puntero, leer el `int`, hacer la operación, escribir de vuelta.

El `list` en sí pesa ~56 bytes (header) + 8 bytes por puntero + el peso de cada `int` (~28 bytes). Para 1 millón de enteros: ~36 MB solo en overhead. Y la aritmética es un bucle en Python interpretado — cada elemento pasa por el bytecode de `BINARY_ADD`, no por una instrucción SIMD de CPU.

### Lo que NumPy hace diferente

Un `np.ndarray` almacena los valores en un **buffer contiguo de memoria** del mismo tipo (`dtype`). Para 1 millón de `float64`: 8 MB exactos, sin punteros, sin objetos por elemento. Y las operaciones se ejecutan en **código C compilado** que vectoriza con instrucciones SIMD (AVX-512 en CPUs modernas) o se descarga a BLAS/LAPACK.

El resultado: una operación como `a + b` con dos arrays de 1M elementos tarda ~1 ms en NumPy contra ~50-200 ms en Python puro. **50-200× de velocidad**, sin contar el ahorro de memoria.

### La regla mnemotécnica

> **Listas**: cuando necesitas heterogeneidad (mezcla de tipos) o tamaños que cambian dinámicamente.
> **NumPy**: cuando los datos son **numéricos, homogéneos y de tamaño fijo**. Esto es el 90% del trabajo en ML.

---

## 2. Conceptual Deep Dive

### 2.1 El array, `shape`, `ndim`, `dtype`

```python
import numpy as np

a = np.array([1, 2, 3, 4])
print(a)             # [1 2 3 4]
print(a.shape)       # (4,)         ← tupla de longitudes por eje
print(a.ndim)        # 1            ← número de ejes (dimensiones)
print(a.dtype)       # int64        ← tipo de cada elemento (fijo, no heterogéneo)
print(a.itemsize)    # 8            ← bytes por elemento
print(a.nbytes)      # 32           ← memoria total del buffer (sin overhead)
```

Para datos tabulares, lo que verás en el 99% de los casos es un array 2D:

```python
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(X.shape)   # (3, 3)  ← 3 filas, 3 columnas
print(X.ndim)    # 2
print(X[0, :])   # [1 2 3] ← primera fila
print(X[:, 0])   # [1 4 7] ← primera columna
```

💡 **Tip mental**: `shape` se lee como `(filas, columnas)` para 2D, `(bloques, filas, columnas)` para 3D, etc. El primer índice es el "más externo" (batch), el último es el "más interno" (features).

### 2.2 Indexing y slicing (vs listas)

```python
a = np.array([10, 20, 30, 40, 50])
a[1:4]           # [20 30 40]    ← igual que listas
a[::-1]          # [50 40 30 20 10]
a[[0, 2, 4]]     # [10 30 50]    ← fancy indexing: pasa una lista de índices

mask = a > 25
a[mask]          # [30 40 50]    ← boolean indexing: filtra por condición
```

Lo que NO existe en listas pero sí en NumPy: el **boolean indexing** (la columna vertebral del filtrado tipo Pandas) y el **fancy indexing** (selección no contigua).

### 2.3 El parámetro `axis`: la pieza más importante

Este es el error #1 en pruebas técnicas. La regla:

- `axis=0` colapsa el **eje 0** (las filas). El resultado tiene una forma con **una fila menos** que el original.
- `axis=1` colapsa el **eje 1** (las columnas). El resultado tiene **una columna menos**.

Concretamente:

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)

X.sum(axis=0)   # [5 7 9]  ← suma por columnas: (1+4, 2+5, 3+6)
X.sum(axis=1)   # [6 15]   ← suma por filas: (1+2+3, 4+5+6)
X.sum()         # 21       ← suma total (sin axis = colapsa todo)
X.mean(axis=0)  # [2.5 3.5 4.5]
X.mean(axis=1)  # [2. 5.]
X.max(axis=0)   # [4 5 6]
```

Mnemotécnica: **`axis=N` es el eje que se ELIMINA**. Si tienes shape `(2, 3)` y haces `axis=0`, te queda `(3,)` — la dimensión 0 desapareció. Si haces `axis=1`, te queda `(2,)`.

⚠️ **Advertencia**: en NumPy, `axis` se pasa por número. En Pandas (lo verás en la siguiente nota), `axis` se pasa por nombre (`index` para filas, `columns` para columnas). Mismo concepto, sintaxis opuesta.

### 2.4 Broadcasting: operar arrays de formas distintas

Broadcasting es la regla que permite hacer `array + escalar` o `array_2d + array_1d` sin copiar memoria. Las dos reglas:

1. Si dos arrays tienen diferente número de dimensiones, el de menor `ndim` se rellena con `1` en su lado izquierdo hasta igualar.
2. En cada dimensión, los tamaños deben ser **iguales** o **uno de ellos debe ser 1**. Si uno es 1, se "estira" (sin copiar memoria) hasta el tamaño del otro.

```python
a = np.array([[1, 2, 3],       # shape (2, 3)
              [4, 5, 6]])
b = np.array([10, 20, 30])    # shape (3,)

a + b                          # broadcasting: b se "estira" a (2, 3)
# [[11 22 33]
#  [14 25 36]]
```

Otro caso típico: restar la media de cada columna.

```python
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
mean = X.mean(axis=0)          # [4. 5. 6.]  shape (3,)
X_centered = X - mean          # broadcasting: (3,3) - (3,) → (3,3)
# [[-3. -3. -3.]
#  [ 0.  0.  0.]
#  [ 3.  3.  3.]]
```

⚠️ **El error clásico**: intentar hacer broadcasting entre `(3,)` y `(2,)` falla con `ValueError: operands could not be broadcast together with shapes (3,) (2,)`. Si dos dimensiones son distintas y ninguna es 1, no hay broadcast posible.

### 2.5 NumPy random: reproducibilidad

```python
np.random.seed(42)                  # fija el estado del generador
np.random.rand(3)                  # [0.37 0.95 0.73]      uniforme [0, 1)
np.random.randn(3)                 # ~N(0, 1)              normal estándar
np.random.randint(0, 10, size=5)   # [3 7 7 0 4]           enteros en [low, high)
np.random.choice([10, 20, 30], size=4, p=[0.5, 0.3, 0.2])  # muestreo con pesos
np.random.permutation(5)           # [2 0 4 1 3]            permutación
```

💡 **Tip**: en pruebas técnicas, **`np.random.seed(42)` antes de generar datos sintéticos** demuestra que sabes que la reproducibilidad importa. Es una línea que suma puntos sin esfuerzo.

### 2.6 Operaciones comunes que verás en la prueba

```python
# Creación
np.zeros((2, 3))           # matriz de ceros
np.ones((2, 3))            # matriz de unos
np.eye(3)                  # matriz identidad 3x3
np.arange(0, 10, 2)        # [0 2 4 6 8]   como range pero devuelve array
np.linspace(0, 1, 5)       # [0.   0.25 0.5 0.75 1. ]   N puntos equiespaciados

# Reducciones (todas aceptan axis=)
np.sum, np.mean, np.std, np.min, np.max, np.median, np.percentile, np.prod, np.cumsum

# Álgebra
np.dot(a, b)               # producto punto (1D) o producto matricial (2D)
a @ b                      # operador equivalente a np.dot
np.linalg.inv(A)           # inversa de matriz
np.linalg.norm(a)          # norma L2 por defecto
np.transpose(X)            # X.T también funciona

# Búsqueda
np.where(a > 5)            # índices donde se cumple la condición
np.argmax(a)               # índice del máximo
np.argsort(a)              # índices que ordenan el array

# Cambiar forma
a.reshape(2, 3)            # vista si es posible, copia si no
a.flatten()                # copia 1D
np.concatenate([a, b])     # concatena a lo largo de axis=0
np.vstack([a, b])          # apila verticalmente
np.hstack([a, b])          # apila horizontalmente
```

---

## 3. Production Reality

### Rendimiento real: el número que convence

```python
import numpy as np, time

N = 1_000_000
lista = list(range(N))
array = np.arange(N)

# Suma en Python puro
t0 = time.perf_counter()
resultado = sum(lista)
t1 = time.perf_counter()
print(f"sum(list):   {(t1-t0)*1000:.1f} ms")

# Suma en NumPy
t0 = time.perf_counter()
resultado = array.sum()
t1 = time.perf_counter()
print(f"np.sum(arr): {(t1-t0)*1000:.1f} ms")
```

Resultado típico en una laptop moderna: **`sum(list) ≈ 8 ms`** vs **`array.sum() ≈ 0.5 ms`**. Una mejora de 16×. Para operaciones más complejas (multiplicación matricial, broadcasting) la diferencia puede ser 100× o más.

### Memory layout: por qué importa para ML

| | Lista de Python | `np.ndarray` |
|---|---|---|
| 1M de `float64` | ~36 MB (overhead) | 8 MB exactos |
| Tipo de los elementos | Heterogéneo (cualquier objeto) | Homogéneo (`dtype` fijo) |
| Localidad en memoria | Puntero a puntero a dato | Contiguo |
| Aritmética | Bucle en bytecode Python | C compilado, vectorizado SIMD |
| Costo de `append` | O(1) amortizado | Crear nuevo array (O(n)) |

⚠️ **Advertencia — `np.append` no es O(1)**: a diferencia de las listas de Python, `np.append` crea un nuevo array y copia. Si necesitas acumular datos, pre-asigna un array del tamaño final y asigna por índice, o usa una lista de Python y conviértela al final.

### Comparativa rápida: cuándo usar NumPy vs stdlib

| Tarea | Mejor herramienta | Por qué |
|---|---|---|
| Media de 5 números | `statistics.mean` o `sum/len` | No vale la pena el overhead de NumPy |
| Media de 1M números | `np.mean` | 50× más rápido que stdlib |
| Distribución normal de 1M muestras | `np.random.randn` | stdlib `random` no tiene normal |
| Multiplicar dos matrices 1000×1000 | `np.dot` o `@` | BLAS optimizado, 1000× más rápido |
| Generar un par de dados | `random.randint` | stdlib es más simple |

---

## 4. Code in Practice

```python
# 📦 NumPy para Análisis de Datos — Cubre: arrays, shape, axis, broadcasting,
# random, reducciones, indexado booleano, comparación de rendimiento.
import numpy as np
import time

# 1. Crear un dataset sintético (3 grupos, 100 muestras cada uno)
np.random.seed(42)
grupo_a = np.random.normal(loc=10, scale=2, size=100)   # media 10, std 2
grupo_b = np.random.normal(loc=15, scale=3, size=100)   # media 15, std 3
grupo_c = np.random.normal(loc=12, scale=1, size=100)   # media 12, std 1

# 2. Apilar en una matriz 2D (filas=muestras, columnas=grupo+valor)
datos = np.column_stack([grupo_a, grupo_b, grupo_c])   # shape (100, 3)
print(f"Shape: {datos.shape}, dtype: {datos.dtype}")

# 3. Reducciones por eje — el caso clásico de prueba
print(f"Media global:    {datos.mean():.2f}")           # todos los 300 valores
print(f"Media por grupo: {datos.mean(axis=0)}")         # [10.0  15.0  12.0]
print(f"Std por grupo:   {datos.std(axis=0)}")          # ~[2.0  3.0  1.0]
print(f"Max por grupo:   {datos.max(axis=0)}")          # max de cada columna

# 4. Indexado booleano (la columna vertebral de Pandas)
valores_altos = datos[datos > 13]                       # filtra todos > 13
print(f"Total valores > 13: {valores_altos.size}")

# 5. Broadcasting: restar la media de cada columna (centrado)
media = datos.mean(axis=0)                              # shape (3,)
centrado = datos - media                                # broadcasting (100,3) - (3,)
print(f"Media post-centrado: {centrado.mean(axis=0)}") # ~[0, 0, 0]

# 6. Normalización min-max (broadcasting sobre el eje 1 = por fila)
mins = datos.min(axis=0)
maxs = datos.max(axis=0)
normalizado = (datos - mins) / (maxs - mins)            # broadcasting (100,3) - (3,)

# 7. argmax, argsort, where
idx_max = np.argmax(datos[:, 1])                        # índice de la fila con mayor valor en grupo B
top_5 = np.argsort(datos[:, 0])[-5:]                    # los 5 índices con mayor valor en grupo A
indices_exactos = np.where(datos[:, 0] > 12)            # índices donde grupo_a > 12

# 8. Rendimiento: lista vs array
N = 1_000_000
lista = list(range(N))
array = np.arange(N)
t0 = time.perf_counter(); s1 = sum(lista); t1 = time.perf_counter()
t2 = time.perf_counter(); s2 = array.sum(); t3 = time.perf_counter()
print(f"sum(list):  {(t1-t0)*1000:6.1f} ms")
print(f"np.sum:     {(t3-t2)*1000:6.1f} ms  ({s1/s2:.0f}× más rápido)")
```

> **Caso real**: en un pipeline de scoring crediticio, los features de un cliente llegan como un array 1D de ~50 valores. Para inferencia, restas la media del training set (broadcasting), divides por la std (broadcasting), multiplicas por los pesos del modelo (`np.dot(features, weights)`) y aplicas una sigmoid. Todo en 4 líneas de NumPy, ~10 µs por cliente. La versión con listas de Python tardaría ~500 µs — 50× más — y eso es la diferencia entre 200 req/s y 10,000 req/s en una sola CPU.

---

## 🎯 Key Takeaways

- **`np.ndarray` almacena datos en buffer contiguo con un `dtype` fijo**, lo que da 16-100× de velocidad y 4-5× menos memoria que listas de Python.
- **`shape` es una tupla de longitudes por eje**; `ndim` es el número de ejes. Para 2D, `shape = (filas, columnas)`.
- **`axis=N` ELIMINA la dimensión N**: `axis=0` colapsa filas (queda `(cols,)`), `axis=1` colapsa columnas (queda `(filas,)`).
- **Broadcasting** permite operar arrays de formas compatibles sin copiar memoria: la regla es que las dimensiones son iguales o una de ellas es 1.
- **`np.random.seed(42)` antes de generar datos** es una buena práctica reproducible — vale puntos en pruebas técnicas.
- **Indexado booleano** (`a[a > 5]`) y **fancy indexing** (`a[[0, 2, 4]]`) no existen en listas; son el pan de cada día en NumPy.
- **`np.dot` / `@` usan BLAS** optimizado. Para matrices grandes, esto es 1000× más rápido que un bucle anidado en Python.

## References

- NumPy Quickstart Tutorial: https://numpy.org/doc/stable/user/quickstart.html
- NumPy Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy Random Generator: https://numpy.org/doc/stable/reference/random/index.html
- From Python to NumPy (book): https://www.labri.fr/perso/nrougier/from-python-to-numpy/
- Related Vault: [[01 - Math y Random]]
- Related Vault: [[05 - Json y Pickle]] (JSON → np.array)
- Related Vault: [[../../05 - Deep Learning y Computer Vision/06 - Computer Vision Pipeline/05 - Computer Vision Pipeline|Computer Vision Pipeline]] (uso real de NumPy)
- Related Vault: [[09 - Pandas para Analisis de Datos|Pandas para Análisis de Datos]] (próxima nota)
- Related Vault: [[../../../projects/01 - Kaggle Competitions - Project Guide|Kaggle Competitions]] (NumPy en competencia)
