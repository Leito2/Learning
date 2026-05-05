# 🖥️ 03 - Entrada y Salida de Datos

Todo programa útil necesita comunicarse con el exterior. En ML/AI Engineering, esto implica leer datasets, guardar modelos serializados y registrar métricas. En Backend, significa responder peticiones HTTP, escribir logs estructurados y procesar argumentos de configuración. Dominar `print`, `input` y el formateo de strings es el primer paso.


## 1. `print()`: Más Allá de lo Básico

La función `print` no solo imprime en pantalla. Sus parámetros permiten control preciso sobre la salida.

```python
print("a", "b", "c")           # a b c (sep por defecto es espacio)
print("a", "b", "c", sep="-")  # a-b-c
print("Sin salto", end="")     # No añade \n al final
print(" continuo")             # continúa en la misma línea
```

| Parámetro | Default | Función |
|-----------|---------|---------|
| `sep` | `" "` | Cadena entre argumentos |
| `end` | `"\n"` | Cadena al final de la impresión |
| `file` | `sys.stdout` | Archivo o stream de salida |
| `flush` | `False` | Fuerza vaciado del buffer si `True` |

⚠️ **Advertencia:** Si rediriges `file` a un archivo de log, usa `flush=True` en logs críticos de errores para evitar pérdida de datos si el programa termina abruptamente.

Caso real: En un sistema de inferencia de ML en tiempo real, se usa `print(json.dumps(prediccion), flush=True)` para enviar resultados línea a línea a un proceso padre a través de un pipe sin buffering.


## 2. Formateo de Strings: Tres Eras

Python ha evolucionado tres mecanismos de formateo. El moderno y recomendado es f-string (Python 3.6+).

### 2.1 Estilo % (heredado)

```python
nombre = "Ana"
edad = 30
print("Hola %s, tienes %d años" % (nombre, edad))
```

### 2.2 `str.format()` (intermedio)

```python
print("Hola {}, tienes {} años".format(nombre, edad))
print("Hola {n}, tienes {e} años".format(n=nombre, e=edad))
```

### 2.3 f-strings (recomendado)

```python
print(f"Hola {nombre}, tienes {edad} años")
print(f"El doble de tu edad es {edad * 2}")  # expresiones inline
print(f"Pi con 4 decimales: {math.pi:.4f}")
print(f"Número alineado: {42:>10}")
```

| Característica | `%` | `.format()` | `f-string` |
|----------------|-----|-------------|------------|
| Legibilidad | Baja | Media | Alta |
| Expresiones inline | No | No | Sí |
| Rendimiento | Lento | Medio | Rápido (evaluación en tiempo de parsing) |
| Recomendado | Legacy | Legacy | ✅ Moderno |

💡 **Tip:** Las f-strings se evalúan en tiempo de definición, no de ejecución repetida. Para logs de alto volumen, precompila plantillas con `.format()` o usa librerías especializadas como `structlog`.


## 3. Formateo Avanzado de Números

```python
valor = 1234.5678

print(f"Entero con signo: {valor:+.2f}")
print(f"Notación científica: {valor:.2e}")
print(f"Porcentaje: {0.85:.1%}")
print(f"Separador de miles: {valor:,.2f}")
print(f"Padding con ceros: {42:08d}")
print(f"Hexadecimal: {255:#x}")
```

Caso real: Un microservicio Backend expone métricas de latencia en formato Prometheus. Usa f-strings para alinear columnas y formatear floats con precisión fija para facilitar el parsing por herramientas de monitoreo.


## 4. `input()` y Casting de Entrada

`input()` siempre devuelve una cadena (`str`). Debes castear explícitamente.

```python
edad_str = input("Introduce tu edad: ")
try:
    edad = int(edad_str)
    print(f"El año que viene tendrás {edad + 1}")
except ValueError:
    print("⚠️ Error: debes introducir un número válido.")
```

⚠️ **Advertencia:** Nunca confíes en la entrada del usuario. Siempre valida y atrapa excepciones (`try/except`) antes de usar los datos en operaciones críticas.


## 5. Codificación UTF-8

Python 3 usa UTF-8 por defecto para archivos fuente y strings en memoria. Esto permite usar emojis, caracteres acentuados y alfabetos no latinos directamente.

```python
texto = "Café 🚀 日本語"
print(texto)
print(f"Longitud en caracteres: {len(texto)}")  # 10 (no bytes)

# Codificar a bytes para transmisión por red
bytes_utf8 = texto.encode("utf-8")
print(f"Bytes: {bytes_utf8}")
print(f"Decodificado: {bytes_utf8.decode('utf-8')}")
```

| Concepto | Descripción |
|----------|-------------|
| `str` | Secuencia de caracteres Unicode (en memoria) |
| `bytes` | Secuencia de bytes (para transmisión/archivos) |
| `encode()` | str → bytes |
| `decode()` | bytes → str |

Caso real: Al consumir una API REST en un Backend, el cuerpo de la respuesta llega como `bytes`. Decodificar con `.decode('utf-8')` es obligatorio antes de parsear JSON con caracteres internacionales.


## 6. Argumentos de Línea de Comandos con `sys.argv`

`sys.argv` es una lista donde el índice 0 es el nombre del script y los siguientes son los argumentos pasados por la terminal.

```python
import sys

print(f"Script: {sys.argv[0]}")
if len(sys.argv) > 1:
    print(f"Primer argumento: {sys.argv[1]}")
else:
    print("No se proporcionaron argumentos.")
```

💡 **Tip:** Para scripts complejos, usa `argparse` (built-in) en lugar de parsear `sys.argv` manualmente. Proporciona `--help` automático y validación de tipos.


## 7. Redirección de `stdout`/`stderr`

Puedes redirigir la salida estándar a archivos o streams personalizados.

```python
import sys

# Guardar stdout original
original_stdout = sys.stdout

with open("salida.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print("Esto va al archivo")

sys.stdout = original_stdout
print("Esto vuelve a la consola")
```

⚠️ **Advertencia:** Modificar `sys.stdout` globalmente puede afectar a librerías de terceros. Usa el parámetro `file` de `print()` siempre que sea posible para un alcance más controlado.


## 8. Caso Real: Logging Básico

```python
import datetime
import sys

def log(mensaje, nivel="INFO", archivo=sys.stdout):
    ahora = datetime.datetime.now().isoformat()
    linea = f"[{ahora}] {nivel:8s} | {mensaje}"
    print(linea, file=archivo, flush=True)

log("Servidor iniciado")
log("Error de conexión", nivel="ERROR", archivo=sys.stderr)
```

En producción se prefiere el módulo `logging` built-in, pero este patrón ilustra el principio: separar niveles de severidad y redirigir errores a `stderr` mientras la información fluye por `stdout`.


## 9. Resumen en Código

```python
# 📦 Código de compresión: Entrada y Salida de Datos
import sys
import math

# 1. print avanzado
print("Valores:", 1, 2, 3, sep=" | ", end="\n---\n")

# 2. f-strings
pi = math.pi
usuario = "Carlos"
print(f"Hola {usuario}, pi={pi:.4f}")

# 3. input con validación
try:
    n = int(input("Número (entero): "))
    print(f"Cuadrado: {n ** 2}")
except ValueError:
    print("Entrada inválida.", file=sys.stderr)

# 4. Codificación
texto = "Datos: ñ, ü, 🚀"
b = texto.encode("utf-8")
print(f"Bytes={b}, decode={b.decode('utf-8')}")

# 5. Argumentos CLI
if len(sys.argv) > 1:
    print(f"Args recibidos: {sys.argv[1:]}")
else:
    print("Sin argumentos CLI.")

# 6. Redirección controlada
with open("resumen_io.txt", "w", encoding="utf-8") as f:
    print("Resumen generado automáticamente.", file=f)
print("✅ Archivo resumen_io.txt creado.")
```
