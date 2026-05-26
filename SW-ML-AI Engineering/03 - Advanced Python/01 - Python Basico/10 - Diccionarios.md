# 🗂️ 10 - Diccionarios

Los diccionarios son el tipo de dato más importante de Python después de las listas. Internamente son tablas hash (hash tables) que ofrecen acceso, inserción y eliminación en tiempo promedio O(1). En ML/AI, mapean nombres de features a valores, almacenan hiperparámetros y gestionan vocabularios. En Backend, representan objetos JSON, cachés, sesiones de usuario y configuraciones.


## 1. Hash Tables Subyacentes

Un diccionario almacena pares clave-valor en una tabla hash. La clave se pasa por una función hash que determina el índice (bucket) donde residirá el valor. Cuando la tabla se llena más allá de un umbral (load factor), Python la redimensiona (rehashing), manteniendo el coste amortizado O(1).

```mermaid
graph LR
    K1["clave 'a'"] -->|hash('a') % 8| B0["Bucket 0"]
    K2["clave 'b'"] -->|hash('b') % 8| B3["Bucket 3"]
    K3["clave 'c'"] -->|hash('c') % 8| B0["Bucket 0<br/>(colisión resuelta)"]
```

💡 **Tip:** CPython 3.6+ implementa diccionarios como tablas hash combinadas con un array índice compacto, lo que mejora la localidad de caché y reduce el uso de memoria.


## 2. Claves Hashables

Para que una clave sea válida, debe ser **hashable**: inmutable durante su vida y con implementación estable de `__hash__` e `__eq__`. Tipos built-in hashables: `int`, `float`, `str`, `bytes`, `tuple` (con elementos hashables), `frozenset`.

```python
# Claves válidas
validas = {
    42: "entero",
    "nombre": "string",
    (1, 2): "tupla",
}

# Clave inválida
# invalida = {[1, 2]: "lista"}  # TypeError: unhashable type: 'list'
```

⚠️ **Advertencia:** Si una clave mutable se modificara después de insertarse, su hash cambiaría y el diccionario no podría encontrarla. De ahí la prohibición de usar listas o diccionarios como claves.


## 3. Métodos Fundamentales

| Método | Función | Complejidad |
|--------|---------|-------------|
| `d[k]` | Acceso por clave (lanza KeyError si no existe) | O(1) |
| `d.get(k, default)` | Acceso seguro con valor por defecto | O(1) |
| `d.keys()` | Vista de claves | O(1) |
| `d.values()` | Vista de valores | O(1) |
| `d.items()` | Vista de pares (clave, valor) | O(1) |
| `d.update(otro)` | Fusiona otro mapping | O(k) |
| `d.pop(k)` | Elimina y devuelve valor | O(1) |
| `d.setdefault(k, v)` | Inserta si clave ausente | O(1) |

```python
usuario = {"nombre": "Ana", "edad": 25}
print(usuario.get("email", "no proporcionado"))  # no proporcionado
usuario.setdefault("rol", "user")
print(usuario)  # {'nombre': 'Ana', 'edad': 25, 'rol': 'user'}
```


## 4. Dict Comprehension

Al igual que las list comprehensions, los diccionarios pueden construirse de forma concisa.

```python
# Cuadrados de 0 a 4
cuadrados = {x: x**2 for x in range(5)}
print(cuadrados)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filtrar pares
pares = {k: v for k, v in cuadrados.items() if v % 2 == 0}
```


## 5. Merging de Diccionarios

Python 3.9+ introduce el operador `|` para fusionar diccionarios. También está disponible `**` en literales.

```python
defaults = {"host": "localhost", "port": 8080}
override = {"port": 3000, "debug": True}

# Python 3.9+
config = defaults | override
print(config)  # {'host': 'localhost', 'port': 3000, 'debug': True}

# Python 3.5+
config2 = {**defaults, **override}
```

| Método | Mutación del original | Requiere versión |
|--------|----------------------|------------------|
| `d1.update(d2)` | Sí (d1 modificado) | Todas |
| `{**d1, **d2}` | No (nuevo dict) | 3.5+ |
| `d1 \| d2` | No (nuevo dict) | 3.9+ |

Caso real: En un servicio de configuración de microservicios, los defaults se almacenan en un diccionario base y los overrides por entorno (dev, staging, prod) se fusionan con `|`, generando configuraciones inmutables y trazables.


## 6. `collections.defaultdict`

`defaultdict` evita `KeyError` proporcionando un valor por defecto automático para claves inexistentes.

```python
from collections import defaultdict

conteo = defaultdict(int)
for letra in "banana":
    conteo[letra] += 1  # No requiere inicialización
print(dict(conteo))  # {'b': 1, 'a': 3, 'n': 2}
```

💡 **Tip:** Usa `defaultdict(list)` para agrupar elementos por categoría sin verificar si la clave ya existe.


## 7. Orden de Inserción (Python 3.7+)

Desde CPython 3.7 (y como parte del lenguaje en 3.7+), los diccionarios preservan el orden de inserción. Esto eliminó la necesidad de `collections.OrderedDict` en la mayoría de casos.

```python
d = {}
d["z"] = 1
d["a"] = 2
d["m"] = 3
print(list(d.keys()))  # ['z', 'a', 'm']
```

⚠️ **Advertencia:** En versiones anteriores a 3.7 (o implementaciones alternativas de Python), no se garantiza el orden. Si tu código debe correr en Python 3.6 o menor, usa `collections.OrderedDict`.


## 8. Uso de Memoria

Los diccionarios priorizan velocidad de acceso sobre eficiencia de memoria. El overhead por clave-valor es significativamente mayor que en estructuras especializadas.

```python
import sys

d = {i: i for i in range(1000)}
print(sys.getsizeof(d))
```

Caso real: Para datasets de vocabulario NLP con millones de tokens, `dict` puede consumir varios GB. Alternativas como `marisa_trie` o arrays NumPy con mapeo inverso reducen drásticamente la memoria.


## 9. Caso Real: Conteo de Frecuencias

```python
from collections import defaultdict

# Conteo de palabras en un corpus
corpus = "el gato y el perro y el gato"
palabras = corpus.split()

frecuencias = defaultdict(int)
for palabra in palabras:
    frecuencias[palabra] += 1

# Top palabras
ordenado = dict(sorted(frecuencias.items(), key=lambda kv: kv[1], reverse=True))
print(ordenado)
```

Este patrón es la base de modelos de bag-of-words y TF-IDF en procesamiento de lenguaje natural.


## 10. Resumen en Código

```python
# 📦 Código de compresión: Diccionarios
from collections import defaultdict

# 1. Creación y acceso
config = {"host": "0.0.0.0", "port": 8080}
print(config.get("debug", False))
config.setdefault("debug", True)

# 2. Merging
override = {"port": 3000}
merged = config | override
print(merged)

# 3. Dict comprehension
squares = {x: x**2 for x in range(6) if x % 2 == 0}
print(squares)

# 4. defaultdict
conteo = defaultdict(int)
for c in "mississippi":
    conteo[c] += 1
print(dict(conteo))

# 5. Iteración
for k, v in merged.items():
    print(f"{k}={v}")

# 6. Orden de inserción
d = {}
d["primero"] = 1
d["segundo"] = 2
print(list(d.keys()))
```
