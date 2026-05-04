# 04 - Async y Await

Las aplicaciones de ML en producción no solo entrenan modelos: sirven predicciones, procesan streams de datos y orquestan múltiples servicios. `async` y `await` permiten manejar miles de conexiones simultáneas sin bloquear el hilo principal.

---

## 1. El problema de la sincronía

```python
import time
import requests

# ❌ SINCRONO: cada petición bloquea hasta recibir respuesta
def obtener_datos(urls):
    resultados = []
    for url in urls:
        respuesta = requests.get(url)  # Bloquea aquí
        resultados.append(respuesta.json())
    return resultados

urls = ["https://api.ejemplo.com/datos/1"] * 10
inicio = time.perf_counter()
obtener_datos(urls)
print(f"Síncrono: {time.perf_counter() - inicio:.2f}s")
```

Si cada petición tarda 0.5s, 10 peticiones tardan **5 segundos**.

---

## 2. Corrutinas con `async` y `await`

Una corrutina es una función que puede suspenderse en `await` y ceder el control para que otras corran.

```python
import asyncio

async def tarea(nombre, duracion):
    print(f"[{nombre}] Iniciando...")
    await asyncio.sleep(duracion)  # Simula I/O sin bloquear
    print(f"[{nombre}] Completada")
    return f"Resultado de {nombre}"

async def main():
    # Ejecutar secuencialmente (await uno por uno)
    r1 = await tarea("A", 1)
    r2 = await tarea("B", 1)
    print(r1, r2)  # Tarda 2 segundos

asyncio.run(main())
```

> 💡 `await` solo puede usarse dentro de funciones `async`. No bloquea el CPU, cede el control al event loop.

### El Event Loop: corazón de async

El **event loop** es un bucle infinito que:

1. Revisa qué tareas están listas para ejecutarse.
2. Ejecuta la siguiente tarea hasta que encuentra un `await`.
3. Cuando una tarea se suspende (`await`), el loop pasa a otra tarea lista.
4. Cuando una operación de I/O termina (ej. respuesta HTTP llega), la tarea suspendida se vuelve lista.

**Analogía:** El event loop es como un chef en un restaurante. En lugar de esperar a que hierva el agua (bloqueo), el chef pone la olla y mientras tanto corta verduras (otra tarea). Cuando el agua hierve, vuelve a la olla.

```python
import asyncio

async def tarea_visual(nombre, pasos):
    for paso in range(pasos):
        print(f"  [{nombre}] paso {paso + 1}/{pasos}")
        await asyncio.sleep(0.1)  # Cede control

async def main():
    # El event loop intercala ambas tareas automáticamente
    await asyncio.gather(
        tarea_visual("A", 3),
        tarea_visual("B", 3),
    )

asyncio.run(main())
```

### Concurrencia vs Paralelismo

| Concepto | Definición | En Python async |
|----------|------------|-----------------|
| **Concurrencia** | Múltiples tareas progresan en un período de tiempo | Async: una sola tarea ejecuta a la vez, pero se alternan rápidamente |
| **Paralelismo** | Múltiples tareas ejecutan simultáneamente | No lo da async solo; necesitas multiprocessing o threads |

> ⚠️ **El GIL (Global Interpreter Lock):** Python tiene un GIL que impide que múltiples threads ejecuten bytecode Python simultáneamente. Por eso, async no acelera tareas CPU-bound. Para CPU-bound, usa `multiprocessing` o `ProcessPoolExecutor`.

---

## 3. Ejecutar múltiples tareas en paralelo

```python
async def main_paralelo():
    # Ejecutar concurrentemente (todas a la vez)
    tareas = [
        tarea("A", 1),
        tarea("B", 1),
        tarea("C", 1),
    ]
    resultados = await asyncio.gather(*tareas)
    print(resultados)  # Tarda ~1 segundo en total

asyncio.run(main_paralelo())
```

| Función | Propósito |
|---------|-----------|
| `asyncio.gather()` | Ejecutar múltiples awaitables y esperar todos |
| `asyncio.create_task()` | Iniciar una tarea en segundo plano |
| `asyncio.wait()` | Esperar con control sobre condiciones (FIRST_COMPLETED, ALL_COMPLETED) |

---

## 4. `async for` y `async with`

### Async for — iteradores asíncronos

Útil para procesar streams de datos (Kafka, WebSockets, colas).

```python
class AsyncDataStreamer:
    """Simula un stream de datos asíncrono."""
    def __init__(self, total):
        self.total = total
        self.actual = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.actual >= self.total:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # Simula latencia de red
        self.actual += 1
        return f"dato_{self.actual}"

async def procesar_stream():
    async for dato in AsyncDataStreamer(5):
        print(f"Procesando: {dato}")

asyncio.run(procesar_stream())
```

### Async with — context managers asíncronos

```python
class AsyncDatabaseConnection:
    async def __aenter__(self):
        print("Conectando a la base de datos...")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("Cerrando conexión...")
        await asyncio.sleep(0.05)

async def consultar():
    async with AsyncDatabaseConnection() as db:
        print("Ejecutando consulta")
        await asyncio.sleep(0.1)

asyncio.run(consultar())
```

---

## 5. Cuándo usar async en ML

| Escenario | ¿Async? | Razón |
|-----------|---------|-------|
| API REST serving modelos | ✅ | Múltiples requests concurrentes |
| Preprocesamiento de imágenes en batch | ❌ | CPU-bound, usa multiprocessing |
| Descarga de datasets desde URLs | ✅ | I/O-bound |
| Entrenamiento de red neuronal | ❌ | GPU-bound, usa PyTorch nativo |
| Streaming de predicciones en tiempo real | ✅ | I/O-bound con WebSockets |

> ⚠️ **Regla de oro:** Async solo acelera operaciones de **I/O** (red, disco, sleep). Para CPU intensivo, usa `multiprocessing` o `ProcessPoolExecutor`.

---

## 6. Mezclar sync y async

Si necesitas llamar código síncrono desde async, usa `loop.run_in_executor()`.

```python
import time

def tarea_pesada_cpu(n):
    """Función síncrona que consume CPU."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

async def main():
    loop = asyncio.get_event_loop()
    # Ejecutar en thread pool para no bloquear el event loop
    resultado = await loop.run_in_executor(None, tarea_pesada_cpu, 10_000_000)
    print(f"Resultado: {resultado}")

asyncio.run(main())
```

---

## 📦 Código de compresión: API de Inferencia con FastAPI + Async

```python
"""
Servidor de inferencia async que procesa múltiples requests
concurrentes sin bloquearse.
Similar a cómo funciona un endpoint de modelos en producción.
"""
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI(title="ML Inference Server")

# Simulación de modelo cargado
class DummyModel:
    def predict(self, data: list) -> float:
        time.sleep(0.1)  # Simula inferencia (CPU-bound en realidad)
        return sum(data) * 0.5

modelo = DummyModel()

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Endpoint async que no bloquea otras requests."""
    loop = asyncio.get_event_loop()

    # La inferencia real es CPU-bound, la movemos a thread pool
    resultado = await loop.run_in_executor(
        None,
        modelo.predict,
        request.features
    )

    return {
        "prediction": resultado,
        "status": "success"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

# Para ejecutar: uvicorn nombre_archivo:app --workers 4

# --- Cliente async para probar concurrencia ---
async def cliente_stress_test():
    import aiohttp

    async with aiohttp.ClientSession() as session:
        tareas = []
        for i in range(20):
            payload = {"features": [i, i+1, i+2]}
            tareas.append(
                session.post("http://localhost:8000/predict", json=payload)
            )

        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*tareas)
        fin = time.perf_counter()

        print(f"20 requests en {fin - inicio:.2f} segundos")
        # Con async: ~0.5s (paralelo)
        # Sin async: ~2.0s (secuencial)

# asyncio.run(cliente_stress_test())
```

---

## 🎯 Proyecto documentado: Servidor de Agentes AI Concurrente

### Descripción
Diseña un servidor async que reciba tareas de múltiples agentes AI, las encole, y las distribuya a workers que invocan diferentes LLMs (OpenAI, Anthropic, local vía Ollama) de forma concurrente. El servidor debe soportar backpressure (limitar requests pendientes) y cancelación de tareas largas.

### Requisitos funcionales
1. **Queue distribuida**: usa `asyncio.Queue` con límite máximo de 100 tareas.
2. **Workers**: 5 workers async que consuman de la cola y llamen a diferentes providers de LLM.
3. **Timeout**: cada invocación a LLM debe tener timeout de 30s.
4. **Backpressure**: si la cola está llena, el endpoint HTTP retorna 503.
5. **Cancelación**: el cliente puede enviar un `task_id` y cancelar la tarea si aún no se procesa.
6. **Streaming**: soportar respuestas tipo Server-Sent Events (SSE) para mostrar progreso.

### Arquitectura
```
Cliente HTTP → Endpoint /invoke → Queue → Worker 1 → OpenAI API
                                    → Worker 2 → Anthropic API
                                    → Worker 3 → Ollama local
                                    → Worker 4 → OpenAI API
                                    → Worker 5 → Ollama local
```

### Métricas de éxito
- Throughput: mínimo 50 requests/segundo con 5 workers.
- Latencia p95 menor a 5s para tareas simples.
- Sin pérdida de tareas ante picos de tráfico.

### Referencias
- FastAPI async patterns
- `asyncio.Queue` documentation
- OpenAI async client (`openai.AsyncOpenAI`)
- SSE (Server-Sent Events) protocol
