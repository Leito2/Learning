# 🔒 09 - Tuplas

Las tuplas son secuencias ordenadas e inmutables. Aunque parecen "listas de solo lectura", su verdadero poder radica en la semántica: una tupla representa un registro heterogéneo con posición fija. En ML/AI, las tuplas almacenan coordenadas (x, y), dimensiones de tensores y pares (feature, label). En Backend, encapsulan registros de bases de datos y configuraciones inmutables.


## 1. Tuplas como Registros / Structs Ligeros

A diferencia de las listas, donde todos los elementos suelen ser homogéneos, las tuplas suelen ser heterogéneas: cada posición tiene un significado semántico distinto.

```python
punto = (10, 20)          # (x, y)
persona = ("Ana", 25, True)  # (nombre, edad, activo)
```

⚠️ **Advertencia:** La inmutabilidad de la tupla no garantiza que sus elementos internos sean inmutables. Una tupla que contiene una lista puede "cambiar" indirectamente si se modifica esa lista.

```python
t = (1, 2, [3, 4])
t[2].append(5)
print(t)  # (1, 2, [3, 4, 5])
```


## 2. Unpacking y Packing

El empaquetado (packing) agrupa valores en una tupla. El desempaquetado (unpacking) los extrae.

```python
# Packing
coords = 10, 20  # Paréntesis opcionales

# Unpacking
x, y = coords
print(x, y)  # 10 20

# Ignorar valores con _
nombre, _, activo = ("Ana", 25, True)
```

💡 **Tip:** El unpacking funciona con cualquier iterable. Es idiomático en Python y más eficiente que acceder por índice.


## 3. `collections.namedtuple`

`namedtuple` crea subclases de tupla con campos nombrados, combinando la ligereza de las tuplas con la legibilidad de los diccionarios.

```python
from collections import namedtuple

Persona = namedtuple("Persona", ["nombre", "edad", "email"])
p = Persona("Luis", 30, "luis@example.com")

print(p.nombre)   # Luis
print(p[1])       # 30 (sigue siendo indexable)
print(p._asdict())  # {'nombre': 'Luis', ...}
```

| Característica | `tuple` | `namedtuple` | `dict` |
|----------------|---------|--------------|--------|
| Memoria | Mínima | Baja | Alta |
| Acceso por nombre | No | Sí | Sí |
| Inmutabilidad | Sí | Sí | No (por defecto) |
| Desempaquetado | Sí | Sí | No directo |

Caso real: Un servicio Backend lee filas de PostgreSQL usando `psycopg2` con `RealDictCursor`. Convertir a `namedtuple` permite acceso por nombre manteniendo inmutabilidad y bajo consumo de memoria en cache local.


## 4. Tuplas como Claves de Diccionario

Por ser inmutables y hashables (si sus elementos lo son), las tuplas pueden usarse como claves de diccionario, algo imposible con listas.

```python
cache = {}
cache[("GET", "/usuarios")] = {"status": 200, "datos": []}
cache[("POST", "/login")] = {"status": 201}

print(cache[("GET", "/usuarios")])
```

⚠️ **Advertencia:** Si la tupla contiene un objeto mutable (como una lista), será unhashable y no podrá usarse como clave.

Caso real: En un sistema de caché de una API GraphQL, la clave de cache se construye como tupla `(query_hash, variables_frozen)` para lookups O(1) sin riesgo de mutación accidental.


## 5. Rendimiento vs Listas

Las tuplas son ligeramente más rápidas que las listas para iterar y acceder por índice porque su inmutabilidad permite optimizaciones en CPython. Además, ocupan menos memoria.

```python
import sys

t = (1, 2, 3)
l = [1, 2, 3]
print(sys.getsizeof(t))  # Menor que lista
print(sys.getsizeof(l))
```

| Operación | Tupla | Lista |
|-----------|-------|-------|
| Creación | ~10-20% más rápida | Base |
| Iteración | Ligeramente más rápida | Base |
| `append`/`pop` | ❌ No permite | ✅ Sí |
| Uso como dict key | ✅ Sí | ❌ No |


## 6. Inmutabilidad Superficial

La inmutabilidad de una tupla se refiere solo a la **estructura de la tupla**, no a los objetos que contiene.

```mermaid
graph TD
    T["Tupla (1, 2, [3])"] --> A["int 1"]
    T --> B["int 2"]
    T --> C["lista [3]"]
    C -->|append(4)| C2["lista [3, 4]"]
    style C fill:#ffcccc
    style C2 fill:#ffcccc
```

💡 **Tip:** Para garantizar inmutabilidad profunda, usa solo tipos inmutables dentro de la tupla (int, str, float, otras tuplas). Si necesitas estructuras anidadas inmutables, considera `frozenset` o tuplas anidadas.


## 7. Return Múltiple de Funciones

Python permite devolver múltiples valores empaquetados en una tupla automáticamente.

```python
def dividir_con_resto(a, b):
    if b == 0:
        raise ValueError("División por cero")
    cociente = a // b
    resto = a % b
    return cociente, resto  # Empaqueta en tupla

c, r = dividir_con_resto(17, 5)
print(f"{c} con resto {r}")  # 3 con resto 2
```

Caso real: Una función de preprocesamiento de datos ML devuelve `(X_train, X_test, y_train, y_test)` como tupla, permitiendo desempaquetado limpio en una sola línea.


## 8. Resumen en Código

```python
# 📦 Código de compresión: Tuplas
from collections import namedtuple

# 1. Tupla como registro
punto = (10, 20)
x, y = punto
print(f"Coordenadas: ({x}, {y})")

# 2. namedtuple
Persona = namedtuple("Persona", "nombre edad")
p = Persona(nombre="Sofia", edad=28)
print(f"{p.nombre} tiene {p.edad} años")

# 3. Tupla como clave de dict
cache = {}
cache[("es", "hola")] = "hello"
print(cache[("es", "hola")])

# 4. Inmutabilidad superficial
t = (1, 2, [3])
t[2].append(4)
print(f"Tupla 'modificada': {t}")

# 5. Return múltiple
def minmax(nums):
    return min(nums), max(nums)

lo, hi = minmax([3, 1, 4, 1, 5])
print(f"min={lo}, max={hi}")

# 6. Comparación de tamaño
import sys
print(f"sizeof tuple(1,2,3): {sys.getsizeof((1,2,3))}")
print(f"sizeof list[1,2,3]: {sys.getsizeof([1,2,3])}")
```
