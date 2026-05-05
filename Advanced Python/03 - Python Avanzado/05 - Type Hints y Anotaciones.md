# 🏷️ 05 - Type Hints y Anotaciones

En proyectos de ML/AI Engineering y Backend, la escalabilidad del código depende en gran medida de su legibilidad y mantenibilidad. Los type hints (PEP 484 en adelante) permiten documentar contratos de funciones, detectar errores antes del runtime y potenciar la autocompletación de IDEs.

![Python Typing](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1024px-Python-logo-notext.svg.png)

---

## 1. Typing Básico

Las anotaciones de tipo se añaden usando la sintaxis de dos puntos `:` para variables y `->` para el retorno de funciones.

```python
def saludar(nombre: str, veces: int = 1) -> str:
    return (f"Hola, {nombre}\n") * veces

edad: int = 30
activo: bool = True
```

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| `int`, `float`, `bool`, `str` | Tipos primitivos. | `x: int = 5` |
| `list`, `dict`, `tuple`, `set` | Colecciones (requieren generics para elementos). | `nums: list[int]` |
| `Optional[T]` | Valor de tipo `T` o `None`. | `nombre: Optional[str]` |
| `Union[T1, T2]` | Uno de varios tipos. | `valor: Union[int, str]` |
| `Any` | Cualquier tipo (deshabilita chequeo). | `dato: Any` |

⚠️ **Advertencia:** Python **no** impone tipos en runtime. Las anotaciones son metadatos. Para verificación estática, se usa `mypy`.

---

## 2. Generics y Colecciones Tipadas

Los generics permiten parametrizar tipos, haciendo el código más expresivo y seguro.

```python
from typing import List, Dict, Tuple, Set

def procesar_batch(datos: List[Dict[str, float]]) -> Tuple[float, float]:
    promedios = [sum(d.values()) / len(d) for d in datos if d]
    if not promedios:
        return 0.0, 0.0
    return min(promedios), max(promedios)

# Uso
batch = [
    {"loss": 0.5, "acc": 0.9},
    {"loss": 0.3, "acc": 0.95}
]
print(procesar_batch(batch))
```

| Generic | Uso típico |
|---------|------------|
| `List[T]` | Lista homogénea de elementos `T`. |
| `Dict[K, V]` | Diccionario con claves `K` y valores `V`. |
| `Tuple[T1, T2, ...]` | Tupla con tipos fijos por posición. |
| `Set[T]` | Conjunto de elementos `T`. |

💡 **Tip:** Desde Python 3.9+, puedes usar los tipos built-in directamente: `list[int]`, `dict[str, float]`, etc., sin importar desde `typing`.

---

## 3. `Callable`, `TypeVar` y `Protocol`

### 3.1 `Callable`

Para tipar funciones que reciben o devuelven otras funciones.

```python
from typing import Callable

def ejecutar_con_log(func: Callable[[int, int], int], a: int, b: int) -> int:
    print(f"Ejecutando {func.__name__}({a}, {b})")
    return func(a, b)

def sumar(a: int, b: int) -> int:
    return a + b

print(ejecutar_con_log(sumar, 2, 3))
```

### 3.2 `TypeVar`

Crea variables de tipo para funciones genéricas.

```python
from typing import TypeVar

T = TypeVar('T')

def primer_elemento(items: list[T]) -> T | None:
    return items[0] if items else None

print(primer_elemento([1, 2, 3]))      # int
print(primer_elemento(["a", "b"]))     # str
```

### 3.3 `Protocol` (Duck Typing Estructural)

Un `Protocol` define una interfaz sin requerir herencia explícita. Si un objeto "camina como pato y habla como pato", es un pato.

```python
from typing import Protocol

class Entrenable(Protocol):
    def entrenar(self, epochs: int) -> None: ...
    def evaluar(self) -> float: ...

def pipeline(modelo: Entrenable) -> None:
    modelo.entrenar(epochs=10)
    print(f"Evaluación: {modelo.evaluar()}")

# No necesita heredar de Entrenable
class RedNeuronal:
    def entrenar(self, epochs: int) -> None:
        pass
    def evaluar(self) -> float:
        return 0.95

pipeline(RedNeuronal())  # OK para mypy
```

Caso real: En backend, un `Protocol` puede definir la interfaz de un repositorio de datos, permitiendo intercambiar implementaciones (SQL, NoSQL, Mock) sin modificar la lógica de negocio.

---

## 4. `NamedTuple` Tipado y `dataclasses`

### 4.1 `NamedTuple`

Tuplas con nombres de campo y tipos.

```python
from typing import NamedTuple

class Punto(NamedTuple):
    x: float
    y: float

    def distancia_origen(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Punto(3.0, 4.0)
print(p.distancia_origen())  # 5.0
```

### 4.2 `dataclasses`

Para clases que principalmente almacenan datos.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class Hiperparametro:
    nombre: str
    valor: float
    rango: List[float] = field(default_factory=list)

hp = Hiperparametro("lr", 0.01, [0.001, 0.01, 0.1])
print(hp)
```

| Característica | `NamedTuple` | `@dataclass` |
|----------------|--------------|--------------|
| Inmutabilidad | Por defecto (`_replace`). | Opcional (`frozen=True`). |
| Valores por defecto | Soportados. | Soportados (`default`, `default_factory`). |
| Herencia | Limitada. | Completa. |
| Memoria | Más eficiente (es una tupla). | Más flexible. |

---

## 5. Verificación Estática con `mypy`

`mypy` es el verificador de tipos estático más popular para Python.

```bash
pip install mypy
mypy mi_modulo.py
```

```python
# Código con error tipográfico intencional para mypy
from typing import List

def duplicar(valores: List[int]) -> List[int]:
    return [v * 2 for v in valores]

resultado = duplicar(["a", "b"])  # mypy reportará error aquí
```

⚠️ **Advertencia:** Un código que pasa `mypy` no está libre de bugs, pero elimina una clase entera de errores por inconsistencia de tipos.

---

## 6. Inspección de Tipos en Runtime

Aunque los tipos no se verifican en runtime, puedes acceder a ellos mediante introspección.

```python
import typing

def ejemplo(a: int, b: str) -> bool:
    return True

hints = typing.get_type_hints(ejemplo)
print(hints)
# {'a': <class 'int'>, 'b': <class 'str'>, 'return': <class 'bool'>}
```

Caso real: Frameworks como FastAPI y Pydantic usan `get_type_hints` para generar automáticamente validaciones de request bodies y documentación OpenAPI.

---

```python
# 📦 Código de compresión: Funciones tipadas con generics y Protocol
from typing import Protocol, TypeVar, List, Dict

T = TypeVar('T')

class Serializable(Protocol):
    def to_dict(self) -> Dict[str, T]: ...

def serializar_lista(items: List[Serializable]) -> List[Dict[str, T]]:
    return [item.to_dict() for item in items]

@dataclass
class Producto:
    id: int
    nombre: str
    precio: float

    def to_dict(self) -> Dict[str, T]:
        return {"id": self.id, "nombre": self.nombre, "precio": self.precio}

if __name__ == "__main__":
    productos = [Producto(1, "Laptop", 999.99)]
    print(serializar_lista(productos))
```
