# 02 - Arrays, Listas y Hash Tables

Estas son las estructuras de datos fundamentales que usas todos los días. Entender cómo funcionan internamente te permite elegir la correcta y optimizar el rendimiento.

---

## 1. Arrays (Arreglos)

Un array es una secuencia contigua de elementos del mismo tipo en memoria. La dirección del elemento `i` se calcula como:

$$\text{dirección}(i) = \text{base} + i \times \text{tamaño_elemento}$$

**Ventajas:**
- Acceso por índice: `O(1)`.
- Localidad espacial: los elementos están juntos en memoria, aprovechan caché CPU.

**Desventajas:**
- Tamaño fijo (en implementaciones estáticas).
- Inserción/eliminación en medio: `O(n)` (hay que desplazar elementos).

### Arrays en Python: `list` vs `numpy.ndarray`

```python
import numpy as np

# Python list: array dinámico de punteros a objetos
py_list = [1, 2, 3, 4]
print(f"Tamaño por elemento: ~28 bytes (puntero + overhead de int)")

# NumPy array: buffer contiguo de tipos primitivos
np_arr = np.array([1, 2, 3, 4], dtype=np.int32)
print(f"Tamaño por elemento: {np_arr.dtype.itemsize} bytes")
print(f"Contiguo en memoria: {np_arr.flags['C_CONTIGUOUS']}")
```

> 💡 **Caso real:** Los tensores de PyTorch son `ndarray`-like. La contigüidad en memoria es crítica para la vectorización GPU.

---

## 2. Listas enlazadas

A diferencia de arrays, las listas enlazadas no requieren contigüidad en memoria. Cada nodo almacena un valor y un puntero al siguiente.

### Singly Linked List

```python
class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.siguiente = None

class LinkedList:
    def __init__(self):
        self.cabeza = None

    def append(self, valor):
        nuevo = Nodo(valor)
        if not self.cabeza:
            self.cabeza = nuevo
            return
        actual = self.cabeza
        while actual.siguiente:
            actual = actual.siguiente
        actual.siguiente = nuevo

    def insert_inicio(self, valor):
        """O(1) - ventaja clave sobre array."""
        nuevo = Nodo(valor)
        nuevo.siguiente = self.cabeza
        self.cabeza = nuevo
```

| Operación | Array | Linked List |
|-----------|-------|-------------|
| Acceso por índice | `O(1)` | `O(n)` |
| Inserción inicio | `O(n)` | `O(1)` |
| Inserción final | `O(1)` amortizado | `O(n)` (sin tail pointer) |
| Inserción medio | `O(n)` | `O(n)` (búsqueda) |
| Memoria | Contigua, eficiente | Fragmentada, overhead por puntero |

> 💡 **En ML:** Las linked lists rara vez se usan directamente, pero los computational graphs de frameworks como PyTorch son esencialmente grafos de nodos conectados por punteros.

---

## 3. Hash Tables (Tablas hash)

Las hash tables son la estructura detrás de los `dict` y `set` de Python. Permiten operaciones de inserción, búsqueda y eliminación en `O(1)` promedio.

![Hash Table](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Hash_table_3_1_1_0_1_0_0_SP.svg/640px-Hash_table_3_1_1_0_1_0_0_SP.svg.png)

### Función hash

Una función hash convierte una clave en un índice del array subyacente:

$$\text{índice} = \text{hash}(\text{clave}) \mod \text{capacidad}$$

### Manejo de colisiones

Cuando dos claves producen el mismo índice, ocurre una **colisión**.

```mermaid
flowchart TD
    A[Clave A] -->|hash(A) % cap| B[Índice 2]
    C[Clave B] -->|hash(B) % cap| B
    B --> D[Bucket 2]
    D -->|chaining| E[Nodo A]
    E --> F[Nodo B]
```

**Chaining (encadenamiento):** Cada bucket es una lista enlazada de elementos que colisionaron.

**Open addressing:** Si el bucket está ocupado, se busca el siguiente disponible (linear probing, quadratic probing, double hashing).

```python
class HashTable:
    def __init__(self, capacidad=16):
        self.capacidad = capacidad
        self.buckets = [[] for _ in range(capacidad)]
        self.size = 0

    def _hash(self, clave):
        return hash(clave) % self.capacidad

    def put(self, clave, valor):
        indice = self._hash(clave)
        bucket = self.buckets[indice]
        for i, (k, v) in enumerate(bucket):
            if k == clave:
                bucket[i] = (clave, valor)  # Actualizar
                return
        bucket.append((clave, valor))
        self.size += 1

    def get(self, clave):
        indice = self._hash(clave)
        for k, v in self.buckets[indice]:
            if k == clave:
                return v
        raise KeyError(clave)

    def delete(self, clave):
        indice = self._hash(clave)
        bucket = self.buckets[indice]
        for i, (k, v) in enumerate(bucket):
            if k == clave:
                del bucket[i]
                self.size -= 1
                return
        raise KeyError(clave)
```

### Factor de carga y redimensionamiento

$$\text{load factor} = \frac{\text{número de elementos}}{\text{capacidad}}$$

Cuando el load factor supera un umbral (típicamente 0.75), se redimensiona el array y se rehashean todos los elementos. Esto mantiene el `O(1)` amortizado.

---

## 4. Aplicaciones en ML

### Feature Hashing (Hashing Trick)

Cuando tienes millones de features categóricas (ej. palabras en NLP), crear un one-hot vector es imposible. El hashing trick mapea cada feature a un índice fijo mediante una función hash.

```python
def feature_hashing(features, n_buckets=1000):
    """Mapea features categóricas a un vector de dimensión fija."""
    vector = np.zeros(n_buckets)
    for feature in features:
        indice = hash(feature) % n_buckets
        vector[indice] += 1  # o += valor del feature
    return vector

# Ejemplo: texto → vector
palabras = ["machine", "learning", "is", "amazing", "machine"]
vector = feature_hashing(palabras, n_buckets=10)
print(vector)
```

> 💡 **Tradeoff:** Colisiones diferentes features pueden colisionar y sumarse. Aumentar `n_buckets` reduce colisiones pero aumenta memoria.

### Diccionarios de vocabulario

En NLP, los tokenizers usan hash tables para mapear tokens a IDs en `O(1)`:

```python
vocab = {"hello": 0, "world": 1, "ml": 2}
token_id = vocab["hello"]  # O(1)
```

---

## 📦 Código de compresión: LRU Cache con Hash Table + Doubly Linked List

```python
"""
LRU Cache implementado con hash table + doubly linked list.
Estructura usada en sistemas de caché, optimizadores de memoria,
y en frameworks de ML para cachear gradientes computados.
"""
class NodoLRU:
    def __init__(self, clave, valor):
        self.clave = clave
        self.valor = valor
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacidad: int):
        self.capacidad = capacidad
        self.cache = {}  # clave → nodo
        self.head = NodoLRU(0, 0)   # Dummy head
        self.tail = NodoLRU(0, 0)   # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, nodo):
        """Desconectar nodo de la lista."""
        prev, nxt = nodo.prev, nodo.next
        prev.next = nxt
        nxt.prev = prev

    def _add(self, nodo):
        """Agregar nodo al frente (más reciente)."""
        nodo.prev = self.head
        nodo.next = self.head.next
        self.head.next.prev = nodo
        self.head.next = nodo

    def get(self, clave):
        if clave in self.cache:
            nodo = self.cache[clave]
            self._remove(nodo)
            self._add(nodo)
            return nodo.valor
        return None

    def put(self, clave, valor):
        if clave in self.cache:
            self._remove(self.cache[clave])
        nodo = NodoLRU(clave, valor)
        self._add(nodo)
        self.cache[clave] = nodo

        if len(self.cache) > self.capacidad:
            # Eliminar el menos reciente (antes de tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.clave]

# --- Uso en ML: cachear embeddings frecuentes ---
cache = LRUCache(capacidad=1000)
for token in corpus:
    embedding = cache.get(token)
    if embedding is None:
        embedding = calcular_embedding(token)  # Costoso
        cache.put(token, embedding)
```

---

## 🎯 Proyecto documentado: Embedding Store con Hash Table y Colisiones Controladas

### Descripción
Diseña un sistema de almacenamiento de embeddings que use una hash table custom con probing lineal para manejar colisiones de forma determinista. El sistema debe soportar millones de embeddings (cada uno de 768 dimensiones float32) con búsqueda en `O(1)` y una tasa de colisión controlada (< 1%).

### Requisitos funcionales
1. `EmbeddingStore(capacity)`: inicializa la tabla hash con una capacidad dada.
2. `insert(key, embedding)`: inserta un embedding asociado a una clave string.
3. `get(key) → embedding`: recupera el embedding en `O(1)` promedio.
4. `delete(key)`: elimina la entrada.
5. `load_factor() → float`: retorna el factor de carga actual.
6. `resize()`: duplica la capacidad y rehashea cuando load_factor > 0.7.
7. Soportar persistencia a disco (guardar/cargar la tabla completa).

### Métricas de éxito
- Búsqueda promedio < 1 microsegundo para 1M embeddings en memoria.
- Colisiones < 1% con load_factor <= 0.7.
- Persistencia: guardar/cargar 1M embeddings en < 5 segundos.

### Referencias
- FAISS (Facebook AI Similarity Search)
- Annoy (Approximate Nearest Neighbors)
- Hash tables en Redis y Memcached
