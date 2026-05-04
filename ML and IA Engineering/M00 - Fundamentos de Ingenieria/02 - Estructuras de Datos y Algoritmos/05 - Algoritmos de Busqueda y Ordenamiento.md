# 05 - Algoritmos de Búsqueda y Ordenamiento

Elegir el algoritmo correcto puede ser la diferencia entre un sistema que funciona y uno que escala. Aquí exploramos los algoritmos fundamentales y sus aplicaciones directas en ML.

---

## 1. Búsqueda lineal

Recorre todos los elementos hasta encontrar el objetivo.

```python
def busqueda_lineal(arr, objetivo):
    for i, elem in enumerate(arr):
        if elem == objetivo:
            return i
    return -1
```

**Complejidad:** `O(n)`.

> 💡 **Cuándo usar:** Datos no ordenados, listas pequeñas, o cuando solo necesitas buscar una vez (el costo de ordenar sería mayor).

---

## 2. Búsqueda binaria

Requiere datos **ordenados**. Divide el espacio de búsqueda a la mitad en cada paso.

```python
def busqueda_binaria(arr, objetivo):
    izq, der = 0, len(arr) - 1
    while izq <= der:
        mid = (izq + der) // 2
        if arr[mid] == objetivo:
            return mid
        elif arr[mid] < objetivo:
            izq = mid + 1
        else:
            der = mid - 1
    return -1
```

**Complejidad:** `O(log n)`.

**Invariante:** Si el objetivo existe, siempre está en `[izq, der]`.

> 💡 **Caso real:** En hyperparameter tuning, si evalúas un rango ordenado de learning rates, puedes usar búsqueda binaria para encontrar el óptimo más rápido que grid search.

### Variante: búsqueda del primer/último

```python
def buscar_primero(arr, objetivo):
    """Encuentra la primera aparición."""
    izq, der = 0, len(arr) - 1
    resultado = -1
    while izq <= der:
        mid = (izq + der) // 2
        if arr[mid] == objetivo:
            resultado = mid
            der = mid - 1  # Seguir buscando a la izquierda
        elif arr[mid] < objetivo:
            izq = mid + 1
        else:
            der = mid - 1
    return resultado
```

---

## 3. Ordenamiento

### Quicksort

Divide y vencerás: elige un pivote, particiona en menores y mayores, recursión.

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivote = arr[len(arr) // 2]
    izq = [x for x in arr if x < pivote]
    medio = [x for x in arr if x == pivote]
    der = [x for x in arr if x > pivote]
    return quicksort(izq) + medio + quicksort(der)
```

**Complejidad:**
- Mejor/promedio: `O(n log n)`.
- Peor caso (pivote malo): `O(n²)`.

**In-place:** La versión clásica usa `O(log n)` espacio para la pila de recursión.

> 💡 **En Python:** `list.sort()` y `sorted()` usan Timsort (híbrido merge + insertion sort), `O(n log n)` garantizado y `O(n)` en casos parcialmente ordenados.

### Mergesort

Divide en mitades, ordena recursivamente, mezcla.

```python
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    izq = mergesort(arr[:mid])
    der = mergesort(arr[mid:])
    return merge(izq, der)

def merge(izq, der):
    resultado = []
    i = j = 0
    while i < len(izq) and j < len(der):
        if izq[i] <= der[j]:
            resultado.append(izq[i])
            i += 1
        else:
            resultado.append(der[j])
            j += 1
    resultado.extend(izq[i:])
    resultado.extend(der[j:])
    return resultado
```

**Complejidad:** `O(n log n)` siempre. **Espacio:** `O(n)`.

> 💡 **Estabilidad:** Mergesort es estable (preserva orden relativo de iguales). Quicksort in-place no lo es por defecto.

### Heapsort

Usa un max-heap para extraer el máximo repetidamente.

**Complejidad:** `O(n log n)` siempre. **Espacio:** `O(1)` adicional.

> 💡 **Ventaja:** Heapsort es in-place y garantiza `O(n log n)`, a diferencia de Quicksort.

### Counting Sort y Radix Sort

Para enteros en un rango acotado, pueden ser `O(n)`.

```python
def counting_sort(arr, max_val):
    conteo = [0] * (max_val + 1)
    for num in arr:
        conteo[num] += 1
    resultado = []
    for i, count in enumerate(conteo):
        resultado.extend([i] * count)
    return resultado
```

> 💡 **Caso real:** Counting sort se usa en histogramas y binning de features numéricas.

---

## 4. Ordenamiento en ML

### Argsort para índices

En ML, frecuentemente necesitas los **índices** ordenados, no los valores.

```python
import numpy as np

probs = np.array([0.1, 0.5, 0.05, 0.35])
indices_ordenados = np.argsort(probs)[::-1]  # [1, 3, 0, 2]
print(f"Top-2 clases: {indices_ordenados[:2]}")
```

### Partial sort (Top-K)

Cuando solo necesitas los `k` mejores, no ordenes todo.

```python
import heapq

# O(n log k) en lugar de O(n log n)
scores = [(0.9, "clase_A"), (0.3, "clase_B"), (0.8, "clase_C"), (0.95, "clase_D")]
top_2 = heapq.nlargest(2, scores)
print(top_2)  # [(0.95, 'clase_D'), (0.9, 'clase_A')]
```

> 💡 **Caso real:** En modelos de lenguaje, para generar texto con top-k sampling, necesitas los k tokens más probables de un vocabulario de 50,000. Usar `nlargest` es mucho más rápido que ordenar todo.

---

## 5. Búsqueda en espacios continuos

En ML, muchos problemas de búsqueda no son discretos.

### Ternary Search

Para funciones unimodales (primero crecen, luego decrecen), divide el espacio en tercios.

```python
def ternary_search(f, izq, der, eps=1e-9):
    """Encuentra máximo de función unimodal."""
    while abs(der - izq) > eps:
        m1 = izq + (der - izq) / 3
        m2 = der - (der - izq) / 3
        if f(m1) < f(m2):
            izq = m1
        else:
            der = m2
    return (izq + der) / 2

# Ejemplo: encontrar learning rate óptimo
# f = lambda lr: evaluar_modelo(lr)
# lr_opt = ternary_search(f, 1e-5, 1.0)
```

> 💡 **Complejidad:** `O(log((b-a)/ε))`.

---

## 📦 Código de compresión: Búsqueda binaria para Threshold Óptimo

```python
"""
Búsqueda binaria para encontrar el threshold de clasificación
que maximiza el F1-score. Útil en modelos de detección de fraude
o diagnóstico médico donde el threshold por defecto (0.5) no es óptimo.
"""
import numpy as np
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_probs):
    """Encuentra threshold que maximiza F1."""
    thresholds = np.sort(np.unique(y_probs))
    mejor_f1 = 0
    mejor_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_thresh = thresh

    return mejor_thresh, mejor_f1

# Optimización con búsqueda binaria sobre F1 como función del threshold
# (asumiendo que F1 es unimodal respecto al threshold, aproximadamente)
def find_optimal_threshold_fast(y_true, y_probs, eps=0.001):
    izq, der = 0.0, 1.0
    mejor_f1 = 0
    mejor_thresh = 0.5

    while der - izq > eps:
        mid = (izq + der) / 2
        # Evaluar en mid - eps, mid, mid + eps
        for thresh in [mid - eps, mid, mid + eps]:
            if 0 <= thresh <= 1:
                y_pred = (y_probs >= thresh).astype(int)
                f1 = f1_score(y_true, y_pred)
                if f1 > mejor_f1:
                    mejor_f1 = f1
                    mejor_thresh = thresh

        # Mover ventana hacia donde F1 es mayor
        y_pred_low = (y_probs >= (mid - eps)).astype(int)
        y_pred_high = (y_probs >= (mid + eps)).astype(int)
        f1_low = f1_score(y_true, y_pred_low)
        f1_high = f1_score(y_true, y_pred_high)

        if f1_low > f1_high:
            der = mid
        else:
            izq = mid

    return mejor_thresh, mejor_f1

# Ejemplo
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_probs = np.array([0.1, 0.2, 0.8, 0.75, 0.9, 0.3, 0.85, 0.15])
thresh, f1 = find_optimal_threshold_fast(y_true, y_probs)
print(f"Threshold óptimo: {thresh:.3f}, F1: {f1:.3f}")
```

---

## 🎯 Proyecto documentado: Sistema de Búsqueda Semántica con FAISS-like Index

### Descripción
Diseña un sistema de búsqueda semántica que indexe millones de documentos usando sus embeddings y permita recuperar los k más similares en tiempo sub-lineal. Implementa un índice Inverted File (IVF) que agrupe embeddings en celdas usando k-means y solo busque en las celdas más cercanas a la query.

### Requisitos funcionales
1. `build_index(embeddings, n_clusters)`: agrupa embeddings con k-means, crea lista invertida por cluster.
2. `search(query_embedding, k)`: encuentra los clusters más cercanos, busca solo en esos clusters, retorna top-k global.
3. `add(new_embedding, doc_id)`: añade nuevo documento al índice.
4. Comparar recall@10 vs búsqueda lineal exacta.
5. Medir tiempo de búsqueda para 1M documentos.

### Métricas de éxito
- Búsqueda de top-10 en < 10ms para 1M documentos de 384D.
- Recall@10 > 90% comparado con búsqueda exacta.
- Tiempo de construcción del índice < 5 minutos.

### Referencias
- FAISS IVF (Inverted File Index)
- Product Quantization
- HNSW (Hierarchical Navigable Small World)
