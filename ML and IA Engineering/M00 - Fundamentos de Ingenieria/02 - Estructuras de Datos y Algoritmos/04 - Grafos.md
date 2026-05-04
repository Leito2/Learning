# 04 - Grafos

Los grafos son la estructura más versátil en ciencias de la computación. En ML, representan desde redes neuronales (computational graphs) hasta redes sociales, knowledge graphs y modelos generativos.

---

## 1. Fundamentos de grafos

Un grafo `G = (V, E)` consiste en:
- **V:** conjunto de vértices (nodos).
- **E:** conjunto de aristas (conexiones entre nodos).

### Tipos de grafos

| Tipo | Característica | Ejemplo en ML |
|------|----------------|---------------|
| **Dirigido** | Las aristas tienen dirección | Computational graphs, DAGs de pipelines |
| **No dirigido** | Aristas bidireccionales | Redes sociales, graphs de co-ocurrencia |
| **Ponderado** | Aristas tienen peso | Graph Attention Networks, PageRank |
| **Cíclico** | Contiene ciclos | RNNs desenrolladas (cadenas) |
| **Acíclico (DAG)** | Sin ciclos | Pipelines de ML, Bayesian networks |
| **Bipartito** | Nodos divididos en dos grupos | Recommender systems (usuarios ↔ items) |

### Representaciones en memoria

**Matriz de adyacencia:** `A[i][j] = 1` si hay arista de `i` a `j`.

- Espacio: `O(V²)`.
- Verificar arista: `O(1)`.
- Listar vecinos: `O(V)`.

**Lista de adyacencia:** Diccionario `nodo → [vecinos]`.

- Espacio: `O(V + E)`.
- Verificar arista: `O(degree)`.
- Listar vecinos: `O(degree)`.

```python
# Lista de adyacencia (recomendada para grafos dispersos)
grafo = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
```

---

## 2. Recorridos de grafos

### BFS (Breadth-First Search)

Explora nivel por nivel, usando una cola. Encuentra el camino más corto en grafos no ponderados.

```python
from collections import deque

def bfs(grafo, inicio):
    visitados = set([inicio])
    cola = deque([inicio])
    orden = []

    while cola:
        nodo = cola.popleft()
        orden.append(nodo)
        for vecino in grafo[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(vecino)
    return orden
```

**Complejidad:** `O(V + E)`.

> 💡 **Caso real:** BFS en un knowledge graph puede encontrar relaciones entre entidades en el menor número de saltos.

### DFS (Depth-First Search)

Explora tan profundo como sea posible antes de retroceder. Usa una pila (explícita o recursiva).

```python
def dfs(grafo, inicio, visitados=None):
    if visitados is None:
        visitados = set()
    visitados.add(inicio)
    print(inicio)
    for vecino in grafo[inicio]:
        if vecino not in visitados:
            dfs(grafo, vecino, visitados)
    return visitados
```

**Complejidad:** `O(V + E)`.

> 💡 **Caso real:** DFS en un computational graph de PyTorch recorre hacia atrás desde la pérdida hasta los parámetros para calcular gradientes (backpropagation es DFS inverso).

### Detección de ciclos

Un grafo dirigido tiene un ciclo si y solo si un nodo se visita dos veces en el mismo camino de DFS.

```python
def tiene_ciclo(grafo):
    ESTADO_BLANCO, ESTADO_GRIS, ESTADO_NEGRO = 0, 1, 2
    estado = {nodo: ESTADO_BLANCO for nodo in grafo}

    def dfs(nodo):
        estado[nodo] = ESTADO_GRIS
        for vecino in grafo[nodo]:
            if estado[vecino] == ESTADO_GRIS:
                return True  # Ciclo detectado
            if estado[vecino] == ESTADO_BLANCO and dfs(vecino):
                return True
        estado[nodo] = ESTADO_NEGRO
        return False

    for nodo in grafo:
        if estado[nodo] == ESTADO_BLANCO:
            if dfs(nodo):
                return True
    return False
```

---

## 3. Caminos más cortos

### Dijkstra

Encuentra el camino más corto desde un nodo a todos los demás en grafos con pesos **no negativos**.

```python
import heapq

def dijkstra(grafo, inicio):
    """
    grafo: dict[nodo] = [(vecino, peso), ...]
    """
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0
    heap = [(0, inicio)]

    while heap:
        dist_actual, nodo = heapq.heappop(heap)
        if dist_actual > distancias[nodo]:
            continue
        for vecino, peso in grafo[nodo]:
            nueva_dist = dist_actual + peso
            if nueva_dist < distancias[vecino]:
                distancias[vecino] = nueva_dist
                heapq.heappush(heap, (nueva_dist, vecino))
    return distancias
```

**Complejidad:** `O((V + E) log V)` con heap.

> 💡 **Caso real:** En un knowledge graph, Dijkstra puede encontrar la relación más "fuerte" (menor peso = mayor confianza) entre dos entidades.

### Bellman-Ford

Maneja pesos negativos y detecta ciclos negativos.

```python
def bellman_ford(grafo, inicio):
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0

    # Relajar V-1 veces
    for _ in range(len(grafo) - 1):
        for nodo in grafo:
            for vecino, peso in grafo[nodo]:
                if distancias[nodo] + peso < distancias[vecino]:
                    distancias[vecino] = distancias[nodo] + peso

    # Detectar ciclos negativos
    for nodo in grafo:
        for vecino, peso in grafo[nodo]:
            if distancias[nodo] + peso < distancias[vecino]:
                raise ValueError("Ciclo negativo detectado")

    return distancias
```

---

## 4. Aplicaciones en ML

### Computational Graphs (PyTorch, TensorFlow)

Cada operación en una red neuronal es un nodo, y los tensores que fluyen son aristas. La backpropagation es un recorrido inverso del grafo.

```python
# Conceptualmente:
# loss = MSE(y_pred, y_true)
# y_pred = Linear(ReLU(Linear(x)))
#
# Grafo:
# x → Linear1 → ReLU → Linear2 → MSE → loss
#   ← grad_w1  ← grad_a  ← grad_w2 ← 1
```

### Knowledge Graphs

Representan entidades y relaciones semánticas:

```
(Barack_Obama) --[born_in]--> (Honolulu)
(Barack_Obama) --[spouse]--> (Michelle_Obama)
(Honolulu) --[located_in]--> (Hawaii)
```

Encontrar caminos en knowledge graphs es clave para question answering y razonamiento.

### Graph Neural Networks (GNNs)

Las GNNs propagan información entre nodos vecinos. La operación fundamental es **message passing**:

$$h_v^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{AGGREGATE}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

Donde `h_v` es el embedding del nodo `v` y `𝒩(v)` son sus vecinos.

> 💡 **Caso real:** Graph Convolutional Networks (GCN) usan el espectro del grafo (eigenvalues de la Laplaciana) para definir convoluciones en grafos.

---

## 📦 Código de compresión: Topological Sort para Pipelines ML

```python
"""
Topological Sort: ordena un DAG de forma que cada nodo
aparece antes que sus dependientes.
Esencial para ejecutar pipelines de ML en el orden correcto.
"""
from collections import deque

def topological_sort(grafo):
    """
    grafo: dict[nodo] = [dependientes]
    Retorna lista ordenada o raise si hay ciclo.
    """
    # Calcular grados de entrada
    in_degree = {nodo: 0 for nodo in grafo}
    for nodo in grafo:
        for dep in grafo[nodo]:
            in_degree[dep] = in_degree.get(dep, 0) + 1

    # Cola con nodos sin dependencias
    cola = deque([n for n, d in in_degree.items() if d == 0])
    orden = []

    while cola:
        nodo = cola.popleft()
        orden.append(nodo)
        for dep in grafo.get(nodo, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                cola.append(dep)

    if len(orden) != len(in_degree):
        raise ValueError("El grafo contiene ciclos")
    return orden

# --- Pipeline de ML como DAG ---
pipeline = {
    "load_data": ["preprocess"],
    "preprocess": ["split", "feature_engineering"],
    "split": ["train"],
    "feature_engineering": ["train"],
    "train": ["evaluate"],
    "evaluate": []
}

orden = topological_sort(pipeline)
print(f"Orden de ejecución: {orden}")
# ['load_data', 'preprocess', 'split', 'feature_engineering', 'train', 'evaluate']
```

---

## 🎯 Proyecto documentado: Graph Neural Network desde Cero

### Descripción
Implementa una Graph Convolutional Network (GCN) simple desde cero usando solo NumPy. El modelo debe aprender embeddings de nodos para clasificación en un grafo de citas (dataset Cora-like).

### Requisitos funcionales
1. Representar el grafo como matriz de adyacencia `A` y matriz de features `X`.
2. Implementar la capa GCN: `H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))`.
   - `Ã = A + I` (self-loops).
   - `D̃` = matriz diagonal de grados.
3. Forward pass con 2 capas GCN.
4. Cross-entropy loss para clasificación de nodos.
5. Backpropagation manual para actualizar `W`.
6. Evaluar accuracy en nodos de test.

### Métricas de éxito
- Accuracy > 80% en dataset sintético de 1000 nodos, 7 clases.
- Tiempo de entrenamiento < 30 segundos en CPU.
- Visualización de embeddings con t-SNE coloreados por clase.

### Referencias
- GCN paper (Kipf & Welling, 2016)
- PyTorch Geometric
- Deep Graph Library (DGL)
