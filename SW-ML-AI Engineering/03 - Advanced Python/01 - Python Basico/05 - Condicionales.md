# 🔄 05 - Condicionales

Los condicionales son la estructura de control más fundamental: permiten que el programa tome decisiones. En ML/AI, se usan para aplicar umbrales en clasificación binaria, validar calidad de datos o seleccionar modelos según métricas. En Backend, implementan reglas de negocio, autorización de rutas y manejo de errores. Escribir condicionales limpios reduce la deuda técnica y previene bugs de lógica.


## 1. `if`, `elif`, `else`

Python evalúa las condiciones de arriba hacia abajo y ejecuta el primer bloque cuya condición sea verdadera. Los bloques se delimitan por indentación (4 espacios recomendados).

```python
nota = 85

if nota >= 90:
    calificacion = "A"
elif nota >= 80:
    calificacion = "B"
elif nota >= 70:
    calificacion = "C"
else:
    calificacion = "F"

print(f"Calificación: {calificacion}")
```

💡 **Tip:** Ordena las condiciones `elif` de la más restrictiva a la menos restrictiva (o viceversa de forma coherente) para evitar que una condición general absorba casos específicos.


## 2. Anidamiento y Legibilidad

Anidar múltiples niveles de `if` produce el "código en forma de pirámide" o "arrow anti-pattern". Cada nivel de indentación aumenta la carga cognitiva.

```python
# ❌ Anidamiento profundo (anti-patrón)
if usuario:
    if usuario.activo:
        if usuario.rol == "admin":
            permitir_acceso()
```

```python
# ✅ Mejor: aplanar con operadores lógicos
if usuario and usuario.activo and usuario.rol == "admin":
    permitir_acceso()
```

⚠️ **Advertencia:** El límite de legibilidad suele estar en 2-3 niveles de anidamiento. Si lo superas, considera extraer funciones o usar guard clauses.


## 3. `match` / `case` (Python 3.10+)

El pattern matching permite desestructurar datos y reemplazar cadenas largas de `if/elif` cuando se compara una variable contra patrones estructurales.

```python
def manejar_comando(comando):
    match comando:
        case "iniciar":
            return "Servicio iniciado"
        case "detener":
            return "Servicio detenido"
        case ["configurar", clave, valor]:
            return f"Configurando {clave}={valor}"
        case {"accion": str(a), "datos": d}:
            return f"Acción {a} con {d}"
        case _:
            return "Comando desconocido"

print(manejar_comando("iniciar"))
print(manejar_comando(["configurar", "host", "localhost"]))
```

| Característica | `if/elif` | `match/case` |
|----------------|-----------|--------------|
| Comparación simple | ✅ Ideal | ✅ Funciona |
| Desestructuración | ❌ Manual | ✅ Nativa |
| Patrones anidados | ❌ Verbosos | ✅ Concisos |
| Requerimiento | Todas las versiones | Python 3.10+ |

Caso real: Un servicio Backend procesa mensajes de una cola (RabbitMQ/Kafka) con distintos formatos JSON. `match/case` permite enrutar cada tipo de mensaje a su handler sin parseo manual repetitivo.


## 4. Operador Ternario

El operador condicional expresivo permite asignar un valor en una sola línea.

```python
edad = 20
estatus = "mayor" if edad >= 18 else "menor"
```

⚠️ **Advertencia:** No anides operadores ternarios. `a if x else b if y else c` es difícil de leer. Usa `if/elif` normales para múltiples ramas.


## 5. `any()` y `all()` en Condiciones

Estas funciones aplican operadores lógicos sobre iterables, facilitando validaciones múltiples.

```python
valores = [5, 8, 12, 3]

if any(v > 10 for v in valores):
    print("Al menos uno supera 10")

if all(v > 0 for v in valores):
    print("Todos son positivos")
```

| Función | Equivalente lógico | Se detiene cuando... |
|---------|-------------------|----------------------|
| `any(iterable)` | `x1 or x2 or ...` | Encuentra el primer `True` |
| `all(iterable)` | `x1 and x2 and ...` | Encuentra el primer `False` |

Caso real: En validación de features de un modelo ML, `all(f is not None for f in features)` garantiza que no hay valores faltantes antes de la inferencia.


## 6. Condicionales con `range` y Membership

`in` verifica pertenencia de forma eficiente (O(1) en sets/dicts, O(n) en listas).

```python
x = 15
if x in range(10, 20):  # 10 <= x < 20
    print("Dentro del rango")

# Para sets (más eficiente para membership testing)
estados_validos = {"activo", "pendiente", "completado"}
if estado in estados_validos:
    procesar()
```

💡 **Tip:** Si haces múltiples verificaciones de pertenencia, convierte la secuencia a `set` para reducir complejidad de O(n) a O(1) por consulta.


## 7. Guard Clauses y Early Returns

Una guard clause es una condición que retorna inmediatamente si los requisitos no se cumplen. Aplana el código y reduce el anidamiento.

```python
def calcular_descuento(precio, cupon):
    if precio is None:
        raise ValueError("Precio requerido")
    if precio < 0:
        raise ValueError("Precio no puede ser negativo")
    if not cupon or not cupon.valido:
        return precio  # early return sin descuento

    return precio * (1 - cupon.porcentaje)
```

Caso real: En una API REST con FastAPI, las guard clauses al inicio de un endpoint validan autenticación, permisos y formato de entrada antes de ejecutar la lógica de negocio, devolviendo errores HTTP 400/401/403 de inmediato.


## 8. Validación de Datos: Caso Real Integrado

```python
def registrar_usuario(nombre, edad, email, roles):
    if not nombre or not isinstance(nombre, str):
        return {"error": "Nombre inválido"}
    if not isinstance(edad, int) or not (18 <= edad <= 120):
        return {"error": "Edad fuera de rango"}
    if "@" not in (email or ""):
        return {"error": "Email inválido"}
    if not roles or not all(isinstance(r, str) for r in roles):
        return {"error": "Roles inválidos"}

    return {"ok": True, "usuario": {"nombre": nombre, "edad": edad, "email": email, "roles": roles}}

print(registrar_usuario("Ana", 25, "ana@example.com", ["admin", "user"]))
print(registrar_usuario("", 25, "ana@example.com", ["user"]))
```


## 9. Resumen en Código

```python
# 📦 Código de compresión: Condicionales
import sys

# 1. if/elif/else
puntaje = 72
if puntaje >= 90:
    letra = "A"
elif puntaje >= 80:
    letra = "B"
elif puntaje >= 70:
    letra = "C"
else:
    letra = "F"
print(f"Letra: {letra}")

# 2. match/case (Python 3.10+)
comando = ["set", "temp", "22"]
match comando:
    case ["set", clave, valor]:
        print(f"Configurar {clave}={valor}")
    case _:
        print("Comando no reconocido")

# 3. Ternario
nivel = "alto" if puntaje > 80 else "bajo"
print(f"Nivel: {nivel}")

# 4. any/all
nums = [2, 4, 6, 8]
print("Todos pares?", all(n % 2 == 0 for n in nums))
print("Alguno > 5?", any(n > 5 for n in nums))

# 5. Guard clauses
def procesar(valor):
    if valor is None:
        return "Error: valor nulo"
    if valor < 0:
        return "Error: negativo"
    return f"Procesado: {valor * 2}"

print(procesar(10))
print(procesar(-5))
```
