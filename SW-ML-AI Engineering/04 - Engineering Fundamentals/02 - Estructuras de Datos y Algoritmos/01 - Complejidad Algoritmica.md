# 01 - Complejidad Algorítmica

Antes de elegir un algoritmo, necesitas saber cuánto tiempo y espacio consumirá cuando los datos crezcan. La complejidad algorítmica es el lenguaje para describir ese crecimiento.

---

## 1. Notación Big-O

Big-O describe el **comportamiento asintótico** de una función: cómo crece cuando la entrada tiende a infinito. Nos importa el término dominante, no las constantes.

$$O(f(n)) = \{g(n) : \exists c, n_0 \text{ tal que } 0 \leq g(n) \leq c \cdot f(n) \text{ para todo } n \geq n_0\}$$

### Jerarquía de complejidades comunes

| Notación | Nombre | Ejemplo en ML |
|----------|--------|---------------|
| `O(1)` | Constante | Acceso a embedding por índice |
| `O(log n)` | Logarítmica | Búsqueda binaria en sorted embeddings |
| `O(n)` | Lineal | Recorrer un dataset una vez |
| `O(n log n)` | Lineal-logarítmica | Merge sort, FFT, algoritmos divide y vencerás |
| `O(n²)` | Cuadrática | Comparar todos los pares de puntos (naive) |
| `O(n³)` | Cúbica | Multiplicación de matrices naive |
| `O(2ⁿ)` | Exponencial | Fuerza bruta en selección de features |
| `O(n!)` | Factorial | Permutaciones de secuencias |

```mermaid
flowchart TD
    A[O(1)] --> B[O(log n)]
    B --> C[O(n)]
    C --> D[O(n log n)]
    D --> E[O(n^2)]
    E --> F[O(n^3)]
    F --> G[O(2^n)]
    G --> H[O(n!)]
```

```python
import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(1, 100, 100)
plt.figure(figsize=(10, 6))
plt.plot(n, np.ones_like(n), label='O(1)')
plt.plot(n, np.log(n), label='O(log n)')
plt.plot(n, n, label='O(n)')
plt.plot(n, n * np.log(n), label='O(n log n)')
plt.plot(n, n**2, label='O(n²)')
plt.plot(n, 2**n, label='O(2ⁿ)')
plt.ylim(0, 500)
plt.xlabel('Tamaño de entrada n')
plt.ylabel('Operaciones')
plt.legend()
plt.title('Crecimiento de funciones')
```

> 💡 **Regla práctica:** Si tu algoritmo es `O(n²)` y procesas 1 millón de items, necesitas ~10¹² operaciones. A 1 GHz, eso son ~1000 segundos ≈ 17 minutos. Si es `O(n log n)`, son ~20 millones de operaciones ≈ 0.02 segundos.

---

## 2. Análisis de algoritmos

```mermaid
flowchart TD
    A[Código] --> B[Bucles anidados]
    B -->|Multiplicar| C[O(n^2)]
    A --> D[Bucles secuenciales]
    D -->|Sumar, queda mayor| E[O(n)]
    A --> F[Recursión]
    F -->|Master Theorem| G[O(n log n)]
```

### Reglas para calcular Big-O

1. **Descartar constantes:** `O(2n) = O(n)`, `O(100) = O(1)`.
2. **Descartar términos de menor orden:** `O(n² + n) = O(n²)`.
3. **Bucles anidados:** multiplicar las complejidades.
4. **Bucles secuenciales:** sumar las complejidades (queda el mayor).
5. **Recursión:** usar el Master Theorem o desenrollar la recurrencia.

### Ejemplo: análisis de código

```python
def encontrar_pares(arr, target):
    """Encuentra todos los pares que suman target."""
    resultado = []
    n = len(arr)
    for i in range(n):           # O(n)
        for j in range(i+1, n):  # O(n) en promedio
            if arr[i] + arr[j] == target:
                resultado.append((arr[i], arr[j]))
    return resultado
# Complejidad total: O(n²)
# Espacio: O(k) donde k = número de pares encontrados
```

### Optimización a O(n) con hash table

```python
def encontrar_pares_optimo(arr, target):
    vistos = set()
    resultado = []
    for num in arr:              # O(n)
        complemento = target - num
        if complemento in vistos:  # O(1) promedio
            resultado.append((complemento, num))
        vistos.add(num)            # O(1) promedio
    return resultado
# Complejidad total: O(n)
# Espacio: O(n)
```

> 💡 **Tradeoff:** Cambiamos tiempo por espacio. De `O(n²)` a `O(n)` gastando `O(n)` de memoria extra.

---

## 3. Complejidad espacial

El espacio también es un recurso crítico, especialmente en ML donde un modelo puede tener miles de millones de parámetros.

| Algoritmo | Tiempo | Espacio | Observación |
|-----------|--------|---------|-------------|
| Merge Sort | `O(n log n)` | `O(n)` | Necesita array auxiliar |
| Quick Sort | `O(n log n)` promedio | `O(log n)` | In-place, espacio para pila de recursión |
| Búsqueda binaria | `O(log n)` | `O(1)` | Solo punteros |
| DFS en grafos | `O(V + E)` | `O(V)` | Pila de recursión o stack explícito |
| BFS en grafos | `O(V + E)` | `O(V)` | Cola de nodos por visitar |

> 💡 **En ML:** Training de LLMs requiere espacio para parámetros + gradientes + optimizador states. Adam guarda `m` y `v` por parámetro → 3x memoria del modelo. Técnicas como LoRA y quantization reducen esto drásticamente.

---

## 4. Análisis amortizado

Algunas operaciones son caras ocasionalmente, pero baratas en promedio cuando se considera una secuencia.

### Ejemplo: ArrayList dinámico

Cuando un array dinámico se llena, se redimensiona (tipicamente duplicando tamaño) y copia todos los elementos. Esa operación individual es `O(n)`, pero **amortizado** cada `append` cuesta `O(1)`.

**Demostración:**
- Inserciones 1 a n: la mayoría son `O(1)`.
- Las redimensiones ocurren en potencias de 2: 1, 2, 4, 8, 16...
- Costo total de redimensiones: `1 + 2 + 4 + ... + n = 2n - 1 = O(n)`.
- Costo amortizado por inserción: `O(n)/n = O(1)`.

```python
class DynamicArray:
    def __init__(self):
        self.data = [None] * 2
        self.size = 0
        self.capacity = 2

    def append(self, value):
        if self.size == self.capacity:
            # Redimensionar: O(n) pero raro
            self._resize(2 * self.capacity)
        self.data[self.size] = value
        self.size += 1  # O(1) amortizado

    def _resize(self, new_cap):
        new_data = [None] * new_cap
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
        self.capacity = new_cap
```

---

## 5. Complejidad en estructuras de Python

| Estructura | Acceso | Búsqueda | Inserción | Eliminación | Notas |
|------------|--------|----------|-----------|-------------|-------|
| `list` | `O(1)` | `O(n)` | `O(1)` amortizado | `O(n)` | Array dinámico |
| `dict` | `O(1)` promedio | `O(1)` promedio | `O(1)` promedio | `O(1)` promedio | Hash table |
| `set` | — | `O(1)` promedio | `O(1)` promedio | `O(1)` promedio | Hash table |
| `collections.deque` | `O(1)` | `O(n)` | `O(1)` | `O(1)` | Doubly-linked list |

> ⚠️ **Peor caso de dict:** `O(n)` si todas las claves colisionan (muy raro con hash bueno).

---

## 📦 Código de compresión: Benchmark de complejidad

```python
"""
Benchmark que demuestra la diferencia práctica entre O(n) y O(n²).
"""
import time
import random

def busqueda_lineal(arr, x):
    """O(n) - recorre todo."""
    for item in arr:
        if item == x:
            return True
    return False

def busqueda_hash(arr, x):
    """O(1) promedio - usa set."""
    conjunto = set(arr)  # O(n) una sola vez
    return x in conjunto  # O(1)

def busqueda_binaria(arr, x):
    """O(log n) - requiere ordenado."""
    izq, der = 0, len(arr) - 1
    while izq <= der:
        mid = (izq + der) // 2
        if arr[mid] == x:
            return True
        elif arr[mid] < x:
            izq = mid + 1
        else:
            der = mid - 1
    return False

# --- Benchmark ---
n = 1_000_000
arr = list(range(n))
x = random.randint(0, n-1)

# Lineal
inicio = time.perf_counter()
busqueda_lineal(arr, x)
print(f"Lineal O(n): {time.perf_counter() - inicio:.4f}s")

# Hash (preprocesamiento + búsqueda)
inicio = time.perf_counter()
conjunto = set(arr)
_ = x in conjunto
print(f"Hash O(1): {time.perf_counter() - inicio:.4f}s")

# Binaria (ya ordenado)
inicio = time.perf_counter()
busqueda_binaria(arr, x)
print(f"Binaria O(log n): {time.perf_counter() - inicio:.4f}s")

# Resultados típicos (n=1M):
# Lineal: ~0.05s
# Hash: ~0.08s (preprocesamiento dominante)
# Binaria: ~0.00001s
```

---

## 🎯 Proyecto documentado: Analizador de Complejidad para Pipelines de ML

### Descripción
Diseña una herramienta que analice automáticamente la complejidad temporal y espacial de un pipeline de ML definido como un grafo de operaciones. Cada nodo del grafo es una operación (ej. "load CSV", "normalize", "fit model") con su complejidad conocida. La herramienta debe calcular la complejidad total del pipeline y detectar cuellos de botella.

### Requisitos funcionales
1. Representar el pipeline como un grafo dirigido acíclico (DAG).
2. Cada nodo tiene: nombre, complejidad temporal `T(n)`, complejidad espacial `S(n)`.
3. Calcular `T_total` y `S_total` del pipeline recorriendo el DAG (considerar que ramas paralelas no suman, toma el máximo).
4. Detectar el nodo con mayor complejidad (cuello de botella).
5. Sugerir optimizaciones: si un nodo es `O(n²)`, proponer alternativa `O(n log n)`.
6. Visualizar el DAG con complejidades anotadas en cada nodo.

### Ejemplo de pipeline
```
[Load Data: O(n)] → [Preprocess: O(n)] → [Feature Engineering: O(n²)] → [Train: O(n·d²)]
                                                          ↓
                                                   [ bottleneck detectado ]
```

### Métricas de éxito
- Análisis correcto de pipelines con hasta 50 nodos.
- Detección de bottleneck con precisión del 100%.
- Sugerencias relevantes basadas en un catálogo de optimizaciones conocidas.

### Referencias
- NetworkX (grafos en Python)
- Big-O cheat sheet
- Profiling de Python (`cProfile`, `line_profiler`)
