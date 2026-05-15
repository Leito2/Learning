# 🧬 Programación Funcional

La programación funcional (PF) trata a las funciones como valores que pueden pasarse, retornarse y combinarse. En el ecosistema de **ML/AI**, las pipelines de transformación de datos de `scikit-learn` están construidas sobre principios funcionales. En **backend**, las funciones de orden superior permiten construir middlewares y decoradores reutilizables sin modificar el código original.

---

## 1. Funciones como objetos de primera clase

En Python, las funciones son objetos. Puedes asignarlas a variables, almacenarlas en estructuras de datos y pasarlas como argumentos.

```python
def saludar(nombre):
    return f"Hola, {nombre}"

def ejecutar(funcion, valor):
    return funcion(valor)

print(ejecutar(saludar, "Mundo"))
```

Caso real: un framework backend como Flask registra funciones como handlers de rutas mediante este mismo principio: `@app.route("/")` almacena la función decorada en un diccionario interno.

---

## 2. Funciones de orden superior

Una función de orden superior es aquella que recibe funciones como parámetros o devuelve funciones.

### 2.1. `map()`

Aplica una función a cada elemento de un iterable.

```python
valores = [1, 2, 3, 4]
cuadrados = list(map(lambda x: x ** 2, valores))
print(cuadrados)  # [1, 4, 9, 16]
```

### 2.2. `filter()`

Selecciona elementos que cumplan una condición.

```python
pares = list(filter(lambda x: x % 2 == 0, valores))
print(pares)  # [2, 4]
```

⚠️ **Advertencia**: en Python 3, `map()` y `filter()` devuelven iteradores. Si necesitas la lista completa más de una vez, materialízala con `list()`.

---

## 3. Funciones lambda

Las lambdas son funciones anónimas de una sola expresión. Son ideales para operaciones simples que no justifican una definición completa.

```python
# Ordenar una lista de tuplas por el segundo elemento
coordenadas = [(1, 5), (3, 2), (0, 8)]
ordenadas = sorted(coordenadas, key=lambda punto: punto[1])
print(ordenadas)  # [(3, 2), (1, 5), (0, 8)]
```

💡 **Tip**: si una lambda ocupa más de una línea o requiere comentarios, es señal de que deberías usar `def`.

---

## 4. `reduce()` de functools

`reduce()` acumula los elementos de un iterable aplicando una función binaria de izquierda a derecha.

```python
from functools import reduce

producto = reduce(lambda x, y: x * y, [1, 2, 3, 4, 5])
print(producto)  # 120 (factorial de 5)
```

Caso real: en procesamiento de datos, `reduce` puede agrupar métricas acumulativas (como suma de pérdidas por batch) en un entrenamiento distribuido sin variables globales.

---

## 5. `sorted()` con clave personalizada

`sorted()` es una función de orden superior que acepta una función `key` para determinar el criterio de ordenamiento.

```python
usuarios = [
    {"nombre": "Ana", "edad": 30},
    {"nombre": "Luis", "edad": 25},
    {"nombre": "Pedro", "edad": 35},
]

por_edad = sorted(usuarios, key=lambda u: u["edad"], reverse=True)
print(por_edad)
```

---

## 6. `partial()` de functools

`partial()` permite fijar un subconjunto de argumentos de una función, creando una nueva función más simple.

```python
from functools import partial

def elevar(base, exponente):
    return base ** exponente

cuadrado = partial(elevar, exponente=2)
cubo = partial(elevar, exponente=3)

print(cuadrado(5))  # 25
print(cubo(3))      # 27
```

Caso real: en un pipeline de ML, puedes crear múltiples normalizadores fijando parámetros como `media` y `desviación` de un dataset de entrenamiento para aplicarlos luego al dataset de validación.

---

## 7. El módulo `operator`

El módulo `operator` proporciona funciones que equivalen a operadores del lenguaje, útiles para evitar lambdas triviales.

```python
from operator import itemgetter, attrgetter, mul
from functools import reduce

# itemgetter para ordenar/dict
datos = [("a", 3), ("b", 1), ("c", 2)]
print(sorted(datos, key=itemgetter(1)))

# attrgetter para objetos
class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

puntos = [Punto(2, 3), Punto(1, 5)]
print(sorted(puntos, key=attrgetter("x")))

# reduce con mul para factorial
print(reduce(mul, range(1, 6), 1))  # 120
```

---

## 8. Comparativa: funcional vs imperativa

| Característica | Imperativa | Funcional |
|----------------|------------|-----------|
| Estado mutable | Sí (variables) | No (preferible inmutable) |
| Bucles | `for`, `while` | Recursión / `map`, `filter`, `reduce` |
| Efectos secundarios | Comunes | Minimizados |
| Legibilidad en Python | Alta para pasos complejos | Alta para pipelines de datos |
| Depuración | Stack traces claros | Más abstracto |

Python no es un lenguaje puramente funcional, pero adoptar un estilo híbrido (funcional para transformaciones de datos, imperativo para control de flujo) es considerado una best practice.

---

## 9. Pipeline funcional completo

```python
from functools import reduce, partial
from operator import itemgetter

# Dataset crudo
registros = [
    {"sensor": "A", "valor": 22.5},
    {"sensor": "B", "valor": 18.0},
    {"sensor": "A", "valor": 23.1},
    {"sensor": "B", "valor": 19.5},
    {"sensor": "A", "valor": 21.8},
]

# Pipeline puramente funcional
pipeline = reduce(
    lambda data, func: func(data),
    [
        partial(filter, lambda r: r["sensor"] == "A"),
        partial(map, lambda r: r["valor"]),
        partial(map, lambda v: round(v, 1)),
        list,
    ],
    registros,
)

print(pipeline)  # [22.5, 23.1, 21.8]
```

---

## 10. Diagrama de pipeline funcional

```mermaid
graph LR
    A[Datos crudos] --> B[filter]</br>condición]
    B --> C[map]</br>extracción]
    C --> D[map]</br>transformación]
    D --> E[list</br>materialización]
    E --> F[Resultado]
```

![Functional Programming](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)

---

## 11. Código de compresión

```python
# Programación Funcional - Esencia
from functools import reduce, partial
from operator import mul, itemgetter

# Pipeline funcional compacto
datos = range(1, 11)
resultado = list(
    map(lambda x: x ** 2,
        filter(lambda x: x % 2 == 0, datos))
)

# reduce
factorial = reduce(mul, range(1, 6), 1)

# partial
base_dos = partial(pow, 2)

# sorted con key
items = [("x", 3), ("y", 1), ("z", 2)]
print(sorted(items, key=itemgetter(1)))

print(resultado, factorial, base_dos(8))
```
