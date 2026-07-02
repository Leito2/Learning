# Async Patterns for Production ML

## 1. Why Async for ML Inference

ML inference is I/O-bound: reading requests from network, loading data from storage, sending results. async/await lets the server handle other requests while waiting for I/O, without the overhead of threads.

## 2. Async Model Server Base

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = await load_model_async()  # non-blocking model load
    yield
    model = None  # cleanup

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(data: dict):
    return await model.predict_async(data["features"])
```

## 3. Async Context Managers for Model Lifecycle

```python
from contextlib import asynccontextmanager

class ModelServer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    @asynccontextmanager
    async def session(self):
        self.model = await self._load()
        try:
            yield self
        finally:
            await self._unload()

    async def _load(self):
        return load_model(self.model_path)  # I/O-bound (disk/s3)

    async def _unload(self):
        del self.model
```

## 4. Async Batch Inference

Collect requests into batches for GPU throughput:

```python
import asyncio

class BatchInference:
    def __init__(self, model, max_batch_size=32, timeout_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout_ms / 1000
        self.queue: list[tuple[dict, asyncio.Future]] = []
        self._task = asyncio.create_task(self._process())

    async def predict(self, data: dict) -> dict:
        future = asyncio.get_event_loop().create_future()
        self.queue.append((data, future))
        return await future

    async def _process(self):
        while True:
            await asyncio.sleep(self.timeout)
            if self.queue:
                batch, futures = zip(*self.queue[:self.max_batch_size])
                self.queue = self.queue[self.max_batch_size:]
                results = self.model.predict([b["features"] for b in batch])
                for future, result in zip(futures, results):
                    future.set_result({"prediction": result.tolist()})
```

## 5. Streaming LLM Responses (SSE)

```python
from fastapi.responses import StreamingResponse
import json

async def stream_predict(prompt: str):
    async for token in model.generate_async(prompt):
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream(prompt: str):
    return StreamingResponse(stream_predict(prompt), media_type="text/event-stream")
```

## 6. Graceful Shutdown

```python
import signal
import asyncio

class Server:
    def __init__(self):
        self._shutdown = asyncio.Event()

    async def run(self):
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown.set)
        await self._shutdown.wait()
        await self.drain_requests()  # finish in-flight requests
        await self.cleanup_model()

    async def drain_requests(self):
        if self.pending_requests > 0:
            await asyncio.sleep(5)  # give time for current requests
```

## 7. Structured Concurrency with anyio

```python
from anyio import create_task_group, run

async def main():
    async with create_task_group() as tg:
        tg.start_soon(serve_requests)
        tg.start_soon(health_check_loop)
        tg.start_soon(metrics_reporter)
    # All tasks complete when any raises or group exits

run(main)
```

## 8. Async Queue for Preprocessing Pipeline

```python
class InferencePipeline:
    def __init__(self, model, max_concurrency=4):
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def process(self, request: dict) -> dict:
        async with self.semaphore:
            features = await self._preprocess(request)
            result = await self._infer(features)
            return await self._postprocess(result)

    async def _preprocess(self, req):
        await asyncio.sleep(0.01)  # simulate I/O
        return req["features"]

    async def _infer(self, features):
        return await self.model.predict_async(features)

    async def _postprocess(self, result):
        return {"prediction": result.tolist()}
```

## Key Takeaways

- Async model server: `asynccontextmanager` for lifecycle, `lifespan` in FastAPI
- Batch inference: asyncio.Queue + Future pattern for GPU utilization
- SSE streaming: `StreamingResponse` + `async for` for LLM tokens
- Graceful shutdown: signal handlers + `asyncio.Event` + drain requests
- Structured concurrency: `anyio.create_task_group` for task lifecycle management
- Semaphore: limit concurrent inference to control GPU memory
