# 06 - Metaclases y Metaprogramación

Las metaclases son el "nivel más profundo" de Python: clases que crean clases. Frameworks como Django ORM, SQLAlchemy y PyTorch usan metaclases para crear APIs declarativas donde defines una clase y el framework genera código automáticamente.

---

## 1. ¿Qué es una metaclase?

En Python, **todo es un objeto**, incluidas las clases. Una clase es instancia de su metaclase.

```python
class MiClase:
    pass

print(type(MiClase))  # <class 'type'>
```

Por defecto, `type` es la metaclase de todas las clases. Puedes crear tu propia metaclase heredando de `type`.

---

## 2. Crear una metaclase

Una metaclase define cómo se construye una clase. Implementa `__new__` y/o `__init__`.

```python
class AutoRegistryMeta(type):
    """Metaclase que registra automáticamente cada clase creada."""
    registry = {}

    def __new__(mcs, name, bases, namespace):
        # mcs = metaclass, name = nombre de la clase, bases = clases padre, namespace = dict de atributos
        cls = super().__new__(mcs, name, bases, namespace)
        mcs.registry[name] = cls
        print(f"Registrada clase: {name}")
        return cls

class ModeloBase(metaclass=AutoRegistryMeta):
    pass

class ResNet(ModeloBase):
    pass

class VGG(ModeloBase):
    pass

print(AutoRegistryMeta.registry)
# {'ModeloBase': <class '__main__.ModeloBase'>, 'ResNet': ..., 'VGG': ...}
```

---

## 3. Modificar clases al nacer

Las metaclases pueden inyectar métodos, validar atributos o transformar definiciones.

```python
class ValidarAtributosMeta(type):
    """Fuerza que todas las clases tengan un atributo 'nombre'."""
    def __new__(mcs, name, bases, namespace):
        if name != "Base" and "nombre" not in namespace:
            raise TypeError(f"{name} debe definir 'nombre'")
        return super().__new__(mcs, name, bases, namespace)

class Base(metaclass=ValidarAtributosMeta):
    pass

class Usuario(Base):
    nombre = "Usuario"  # OK

# class Producto(Base):
#     pass  # TypeError: Producto debe definir 'nombre'
```

---

## 4. `__new__` vs `__init__` en metaclases

| Método | Cuándo se llama | Uso típico |
|--------|-----------------|------------|
| `__new__` | Antes de crear la clase | Modificar namespace, validar, registrar |
| `__init__` | Después de crear la clase | Configuración post-creación |

```python
class MetaEjemplo(type):
    def __new__(mcs, name, bases, namespace):
        print(f"__new__ creando {name}")
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        print(f"__init__ inicializando {name}")
        super().__init__(name, bases, namespace)

class Ejemplo(metaclass=MetaEjemplo):
    pass
# __new__ creando Ejemplo
# __init__ inicializando Ejemplo
```

---

## 5. Descriptores: control de atributos

Un descriptor es un objeto que implementa `__get__`, `__set__` o `__delete__`. Es la base de `property`, `classmethod`, `staticmethod`.

```python
class ValidadorRango:
    """Descriptor que valida que un atributo esté en un rango."""
    def __init__(self, minimo, maximo):
        self.minimo = minimo
        self.maximo = maximo
        self.nombre = None

    def __set_name__(self, owner, name):
        self.nombre = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.nombre]

    def __set__(self, instance, value):
        if not (self.minimo <= value <= self.maximo):
            raise ValueError(
                f"{self.nombre} debe estar entre {self.minimo} y {self.maximo}"
            )
        instance.__dict__[self.nombre] = value

class Configuracion:
    epochs = ValidadorRango(1, 1000)
    learning_rate = ValidadorRango(0.0001, 1.0)

    def __init__(self):
        self.epochs = 10
        self.learning_rate = 0.001

config = Configuracion()
config.epochs = 50        # OK
# config.epochs = 2000    # ValueError
```

> 💡 **Caso real:** SQLAlchemy usa descriptores para que `usuario.nombre` traduzca a `SELECT nombre FROM usuarios`.

---

## 6. `__getattr__` y `__getattribute__`

Controlan qué ocurre cuando accedes a un atributo.

```python
class LazyModel:
    """Carga pesos solo cuando se accede a ellos."""
    def __init__(self):
        self._pesos_cargados = False

    def __getattr__(self, nombre):
        if nombre == "pesos" and not self._pesos_cargados:
            print("Cargando pesos desde disco...")
            self.pesos = [0.1, 0.2, 0.3]  # Simulación
            self._pesos_cargados = True
            return self.pesos
        raise AttributeError(f"'{type(self).__name__}' no tiene '{nombre}'")

modelo = LazyModel()
print(modelo.pesos)  # Cargando pesos desde disco...
print(modelo.pesos)  # Ya cargados, sin mensaje
```

> ⚠️ `__getattribute__` se llama para **todos** los atributos; `__getattr__` solo cuando el atributo no existe.

---

## 7. Metaprogramación con decoradores de clase

Puedes lograr mucho sin metaclases usando decoradores de clase (más simples).

```python
def singleton(cls):
    """Decorador que convierte una clase en singleton."""
    instancia = {}
    def wrapper(*args, **kwargs):
        if cls not in instancia:
            instancia[cls] = cls(*args, **kwargs)
        return instancia[cls]
    return wrapper

@singleton
class ConfiguracionGlobal:
    def __init__(self):
        self.debug = False

c1 = ConfiguracionGlobal()
c2 = ConfiguracionGlobal()
print(c1 is c2)  # True
```

> 💡 Regla: si puedes resolverlo con un decorador de clase, no uses metaclases. Las metaclases son para cuando necesitas controlar la **creación** de la clase misma.

### El modelo de objetos de Python (MRO y `super`)

Python usa el algoritmo **C3 Linearization** para calcular el Method Resolution Order (MRO). Determina el orden en que Python busca métodos en la jerarquía de herencia.

```python
class A:
    def metodo(self):
        print("A")

class B(A):
    def metodo(self):
        print("B")
        super().metodo()

class C(A):
    def metodo(self):
        print("C")
        super().metodo()

class D(B, C):
    def metodo(self):
        print("D")
        super().metodo()

d = D()
d.metodo()
# D → B → C → A (orden MRO)
print(D.__mro__)
# (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

> 💡 **`super()` no llama al padre directo**, llama al **siguiente en el MRO**. Esto permite patrones de herencia cooperativa (cooperative multiple inheritance).

### `__prepare__` — Controlar el namespace de la clase

Antes de que el cuerpo de la clase se ejecute, Python llama a `__prepare__` para crear el diccionario donde vivirán los atributos. Puedes devolver un `OrderedDict`, un `defaultdict`, o incluso un objeto personalizado.

```python
class OrderedMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        print(f"Preparando namespace para {name}")
        return {}  # Podrías devolver OrderedDict para preservar orden

class Ejemplo(metaclass=OrderedMeta):
    a = 1
    b = 2
```

### `__init_subclass__` — Alternativa moderna a metaclases

Desde Python 3.6, `__init_subclass__` permite ejecutar código cuando una clase es subclasificada, sin necesidad de metaclases.

```python
class PluginBase:
    _plugins = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._plugins.append(cls)
        print(f"Plugin registrado: {cls.__name__}")

class PluginA(PluginBase):
    pass

class PluginB(PluginBase):
    pass

print(PluginBase._plugins)  # [<class '__main__.PluginA'>, <class '__main__.PluginB'>]
```

> 💡 **Recomendación:** Usa `__init_subclass__` antes que metaclases siempre que sea posible. Es más simple y suficiente para la mayoría de los casos.

---

## 📦 Código de compresión: Mini ORM con metaclases

```python
"""
Mini ORM que usa metaclases para registrar campos y generar SQL.
Demuestra cómo Django y SQLAlchemy funcionan internamente.
"""
from typing import Any, Dict, List

class Campo:
    def __init__(self, tipo: str, nullable: bool = True):
        self.tipo = tipo
        self.nullable = nullable

class ModelMeta(type):
    """Metaclase que escanea atributos y registra solo los campos."""
    def __new__(mcs, name, bases, namespace):
        campos = {}
        for key, value in namespace.items():
            if isinstance(value, Campo):
                campos[key] = value

        # Crear la clase
        cls = super().__new__(mcs, name, bases, namespace)
        cls._campos = campos
        cls._nombre_tabla = name.lower()
        return cls

class ModeloBase(metaclass=ModelMeta):
    _campos: Dict[str, Campo] = {}
    _nombre_tabla: str = ""

    def __init__(self, **kwargs):
        for nombre, campo in self._campos.items():
            valor = kwargs.get(nombre)
            if valor is None and not campo.nullable:
                raise ValueError(f"{nombre} no puede ser nulo")
            setattr(self, nombre, valor)

    def insert_sql(self) -> str:
        columnas = ", ".join(self._campos.keys())
        valores = ", ".join(f"'{getattr(self, c)}'" for c in self._campos)
        return f"INSERT INTO {self._nombre_tabla} ({columnas}) VALUES ({valores});"

# --- Uso ---
class Usuario(ModeloBase):
    id = Campo("INTEGER", nullable=False)
    nombre = Campo("VARCHAR", nullable=False)
    email = Campo("VARCHAR", nullable=True)

u = Usuario(id=1, nombre="Ana", email="ana@mail.com")
print(u.insert_sql())
# INSERT INTO usuario (id, nombre, email) VALUES ('1', 'Ana', 'ana@mail.com');

print(Usuario._campos)
# {'id': <__main__.Campo>, 'nombre': <__main__.Campo>, 'email': <__main__.Campo>}
```

---

## 🎯 Proyecto documentado: Framework Declarativo para Capas de Red Neuronal

### Descripción
Diseña un framework declarativo similar a Keras `Sequential` usando metaclases. El usuario define una clase que hereda de `Network` y declara capas como atributos de clase. La metaclase debe construir el grafo computacional, validar compatibilidad de dimensiones entre capas, y generar un método `forward()` automáticamente.

### Requisitos funcionales
1. `Layer`: clase base con `input_dim`, `output_dim`, y método `forward(x)`.
2. `NetworkMeta`: metaclase que:
   - Escanea atributos de tipo `Layer`.
   - Verifica que `output_dim` de capa N coincida con `input_dim` de capa N+1.
   - Genera dinámicamente un método `forward()` que encadena las capas.
3. `Network`: clase base para que el usuario defina modelos.
4. Validación en tiempo de definición de clase (no en runtime).
5. Soporte para `summary()` que imprima arquitectura tipo Keras.

### Ejemplo de uso esperado
```python
class MiRed(Network):
    capa1 = Dense(input_dim=784, output_dim=256, activation="relu")
    capa2 = Dense(input_dim=256, output_dim=10, activation="softmax")

modelo = MiRed()
modelo.summary()
# Layer (type)        Output Shape    Param #
# dense_1 (Dense)     (None, 256)     200960
# dense_2 (Dense)     (None, 10)      2570
```

### Métricas de éxito
- Error claro si dimensiones no coinciden al definir la clase.
- `forward()` generado sin overhead medible.
- Extensible: el usuario puede definir sus propias capas heredando `Layer`.

### Referencias
- Keras `Sequential` source code
- PyTorch `nn.Module` metaclass (`type.__call__` override)
- Python `__init_subclass__` hook
