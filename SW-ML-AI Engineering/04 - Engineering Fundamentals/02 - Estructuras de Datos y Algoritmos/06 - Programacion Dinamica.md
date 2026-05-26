# 06 - Programación Dinámica

La programación dinámica (DP) es una técnica para resolver problemas complejos dividiéndolos en subproblemas superpuestos, resolviendo cada subproblema solo una vez y almacenando sus resultados.

---

## 1. Principios fundamentales

### Dos ingredientes

1. **Subestructura óptima:** La solución óptima del problema contiene soluciones óptimas de subproblemas.
2. **Subproblemas superpuestos:** El mismo subproblema se resuelve múltiples veces en un enfoque naive.

### Enfoques

| Enfoque | Descripción | Espacio |
|---------|-------------|---------|
| **Top-down (memoización)** | Recursión + caché de resultados | `O(n)` stack + memo |
| **Bottom-up (tabulación)** | Iterativo, llenando tabla desde casos base | `O(n)` tabla |

---

## 2. Fibonacci: el ejemplo clásico

### Naive (exponencial)

```mermaid
graph TD
    F5[fib(5)] --> F4[fib(4)]
    F5 --> F3a[fib(3)]
    F4 --> F3b[fib(3)]
    F4 --> F2a[fib(2)]
    F3a --> F2b[fib(2)]
    F3a --> F1a[fib(1)]
    F3b --> F2c[fib(2)]
    F3b --> F1b[fib(1)]
    F2a --> F1c[fib(1)]
    F2a --> F0a[fib(0)]
    F2b --> F1d[fib(1)]
    F2b --> F0b[fib(0)]
    F2c --> F1e[fib(1)]
    F2c --> F0c[fib(0)]
```

```python
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
# Complejidad: O(2^n)
```

### Memoización (top-down)

```python
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
# Complejidad: O(n) tiempo, O(n) espacio
```

### Tabulación (bottom-up)

```python
def fib_tab(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
# Complejidad: O(n) tiempo, O(n) espacio
```

### Optimización de espacio

```python
def fib_optimo(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
# Complejidad: O(n) tiempo, O(1) espacio
```

> 💡 **Lección:** A veces solo necesitas los últimos `k` estados, no toda la tabla.

---

## 3. Problemas clásicos de DP

```mermaid
flowchart TD
    A[Subestructura óptima] --> B[Subproblemas superpuestos]
    B --> C{Enfoque}
    C -->|Top-down| D[Memoización]
    C -->|Bottom-up| E[Tabulación]
    D --> F[O(n) stack + memo]
    E --> G[O(n) tabla]
```

### Knapsack (Mochila)

Dados `n` items con pesos `w[i]` y valores `v[i]`, y una mochila de capacidad `W`, maximiza el valor total sin exceder el peso.

**Definición recursiva:**

```
dp[i][w] = máximo valor usando los primeros i items con capacidad w
```

**Transición:**

$$dp[i][w] = \max(dp[i-1][w], \ v[i] + dp[i-1][w - w[i]])$$

```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]  # No tomar item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                               values[i-1] + dp[i-1][w - weights[i-1]])
    return dp[n][W]
```

**Optimización de espacio:** `dp[w]` en lugar de `dp[i][w]` (iterar `w` de derecha a izquierda).

> 💡 **Caso real:** En feature selection, puedes verlo como un knapsack donde "peso" es complejidad computacional y "valor" es ganancia de performance.

### Longest Common Subsequence (LCS)

Dadas dos secuencias, encuentra la subsecuencia común más larga.

```python
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

> 💡 **Caso real:** En BioNLP, LCS se usa para alinear secuencias de ADN/proteínas. En NLP, para detectar plagio o similitud entre documentos.

### Edit Distance (Levenshtein)

Mínimo número de operaciones (insertar, eliminar, reemplazar) para transformar una cadena en otra.

```python
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # eliminar
                                   dp[i][j-1],    # insertar
                                   dp[i-1][j-1])  # reemplazar
    return dp[m][n]
```

> 💡 **Caso real:** Correctores ortográficos, fuzzy string matching, y evaluación de modelos de traducción automática (WER - Word Error Rate).

---

## 4. DP en ML: Viterbi y HMMs

### Hidden Markov Models (HMM)

Un HMM modela secuencias donde el estado subyacente es "oculto" y solo observamos emisiones.

**Problema de decodificación:** Dada una secuencia de observaciones, encontrar la secuencia de estados más probable.

### Algoritmo de Viterbi

Viterbi usa DP para encontrar el camino más probable en un HMM.

```python
def viterbi(obs, estados, start_prob, trans_prob, emit_prob):
    """
    obs: secuencia de observaciones
    estados: lista de estados posibles
    start_prob: P(estado inicial)
    trans_prob: P(estado_t | estado_t-1)
    emit_prob: P(observación | estado)
    """
    T = len(obs)
    N = len(estados)

    # dp[t][i] = probabilidad máxima de estar en estado i en tiempo t
    dp = [[0.0] * N for _ in range(T)]
    backpointer = [[None] * N for _ in range(T)]

    # Inicialización
    for i in range(N):
        dp[0][i] = start_prob[i] * emit_prob[i][obs[0]]

    # Recursión
    for t in range(1, T):
        for j in range(N):
            max_prob = 0.0
            best_prev = None
            for i in range(N):
                prob = dp[t-1][i] * trans_prob[i][j] * emit_prob[j][obs[t]]
                if prob > max_prob:
                    max_prob = prob
                    best_prev = i
            dp[t][j] = max_prob
            backpointer[t][j] = best_prev

    # Backtracking
    mejor_final = max(range(N), key=lambda i: dp[T-1][i])
    camino = [mejor_final]
    for t in range(T-1, 0, -1):
        camino.append(backpointer[t][camino[-1]])

    return camino[::-1]
```

**Complejidad:** `O(T · N²)` donde `T` = longitud de secuencia, `N` = número de estados.

> 💡 **Caso real:** Reconocimiento de voz (estados = fonemas, observaciones = features acústicos). POS tagging en NLP (estados = etiquetas gramaticales, observaciones = palabras).

---

## 5. DP en optimización de modelos

### Memoización en transformers

En modelos de lenguaje, se puede cachear (memoizar) los key-value pairs de atención para reutilizarlos en generación autoregresiva:

```python
# KV-cache en generación de texto
# En lugar de recalcular atención sobre toda la secuencia en cada paso,
# solo calculas atención sobre el nuevo token usando K y V acumulados.
# Esto reduce complejidad de O(n²) a O(n) por token generado.
```

---

## 📦 Código de compresión: DP para Secuencia de Máxima Suma

```python
"""
Maximum Subarray (Kadane's Algorithm).
Encuentra la subsecuencia contigua con máxima suma.
Útil en análisis de series temporales y detección de anomalías.
"""
def max_subarray_kadane(arr):
    """
    dp[i] = máxima suma de subarray terminando en i
    Transición: dp[i] = max(arr[i], arr[i] + dp[i-1])
    """
    max_actual = max_global = arr[0]
    inicio = fin = inicio_temp = 0

    for i in range(1, len(arr)):
        if arr[i] > max_actual + arr[i]:
            max_actual = arr[i]
            inicio_temp = i
        else:
            max_actual += arr[i]

        if max_actual > max_global:
            max_global = max_actual
            inicio = inicio_temp
            fin = i

    return max_global, arr[inicio:fin+1]

# Ejemplo: detectar período de mayor crecimiento en métricas
metricas = [1, -2, 3, 5, -1, 2, -8, 4, 6]
max_suma, subarray = max_subarray_kadane(metricas)
print(f"Máxima suma: {max_suma}")  # 9 (subarray [3, 5, -1, 2])
print(f"Subarray: {subarray}")
```

---

## 🎯 Proyecto documentado: Alineación de Secuencias para BioNLP

### Descripción
Implementa el algoritmo Needleman-Wunsch para alineación global de secuencias de ADN/proteínas usando programación dinámica. Extiende el algoritmo básico para soportar: gap penalties afines (abrir gap vs extender gap), matrices de scoring (BLOSUM para proteínas), y traceback para reconstruir la alineación óptima.

### Requisitos funcionales
1. `needleman_wunsch(seq1, seq2, match, mismatch, gap)`: alineación global básica.
2. Extensión con gap penalty afín: `gap_open` (penalización alta) y `gap_extend` (penalización baja).
3. Soporte para matrices de scoring externas (PAM, BLOSUM).
4. Reconstrucción de la alineación óptima via traceback.
5. Visualización de la matriz DP con el camino óptimo resaltado.
6. Cálculo de identidad y similitud entre secuencias alineadas.

### Métricas de éxito
- Alineación correcta para secuencias de hasta 5000 caracteres en < 2 segundos.
- Identidad calculada correctamente vs implementaciones de referencia (Biopython).
- Penalización afín produce gaps más largos y continuos (biológicamente realista).

### Referencias
- Needleman-Wunsch algorithm
- Smith-Waterman (alineación local)
- Biopython `PairwiseAligner`
- BLOSUM62 matrix
