# 🛠️ Instalación y Primer Servidor

Este módulo lleva a vLLM de la teoría al primer servidor funcional. Aprenderás a preparar el entorno, elegir un modelo, lanzar `vllm serve` con la configuración correcta y validar el endpoint con curl y el SDK de OpenAI. Es el módulo más "práctico" del curso: al final tendrás un servidor HTTP de inferencia LLM corriendo en tu GPU.

---

## 1. Requisitos de hardware

### 1.1 GPU

vLLM está optimizado para **NVIDIA CUDA**. El soporte AMD ROCm existe pero requiere flags especiales y modelos validados.

| GPU | Memoria | Soporte FP8 | Soporte vLLM | Notas |
|-----|--------:|:-----------:|:------------:|-------|
| H100 SXM/PCIe | 80 GB | ✅ | Excelente | Ideal para producción |
| H200 | 141 GB | ✅ | Excelente | Top tier |
| A100 40GB/80GB | 40/80 GB | ❌ | Excelente | Estándar de producción |
| L40S | 48 GB | ✅ | Excelente | Inferencia optimizada |
| L4 | 24 GB | ❌ | Bueno | Small models |
| RTX 4090 | 24 GB | ❌ | Bueno | Best consumer |
| RTX 3090 | 24 GB | ❌ | Bueno | Soporte comunidad |
| A10 | 24 GB | ❌ | Bueno | Buen balance |
| V100 | 16/32 GB | ❌ | Funciona | Sin FP8, sin FlashAttention-2/3 |
| T4 | 16 GB | ❌ | Limitado | Solo modelos pequeños, FP16 |

> **Regla práctica**: el modelo en FP16 o BF16 ocupa $N_{\text{params}} \times 2$ bytes. Para un 7B necesitas 14 GB solo del modelo. KV cache y overhead añaden ~20%. Una RTX 4090 (24 GB) puede con un 7B FP16 con buen throughput.

### 1.2 Drivers y CUDA

```bash
# Verifica driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# >= 525 para CUDA 12.x

# Verifica CUDA
nvcc --version  # >= 12.1
```

Si tienes una versión antigua, actualiza con el [runfile de NVIDIA](https://www.nvidia.com/drivers) o tu package manager.

### 1.3 CPU y RAM

vLLM descarga el modelo en RAM antes de cargarlo en GPU. Recomendaciones:

| Modelo | RAM mínima | RAM recomendada |
|--------|-----------|-----------------|
| 7B | 16 GB | 32 GB |
| 13B | 24 GB | 48 GB |
| 70B | 80 GB | 160 GB |
| 405B (FP8) | 200 GB | 384 GB |

> **Importante**: en producción con Kubernetes, define el `requests.memory` del pod según el tamaño del modelo + buffer.

### 1.4 Almacenamiento

- **Modelos en HuggingFace Hub**: descargan ~10-200 GB según tamaño. SSD NVMe recomendado.
- **Espacio de swap**: si activas `--preemption-mode swap`, necesitas CPU RAM equivalente al KV cache en uso.

---

## 2. Instalación

### 2.1 Opción 1: pip (la más simple)

```bash
python -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip
pip install vllm
```

Esto instala vLLM con todas las dependencias CUDA. Elige wheels precompilados para tu versión de CUDA.

> **Importante**: no instales `torch` por separado antes. vLLM instala la versión correcta de `torch` como dependencia.

### 2.2 Opción 2: Docker (recomendado para producción)

```bash
docker pull vllm/vllm-openai:latest
```

Imagen oficial con CUDA, vLLM y todas las deps. Para AMD ROCm:

```bash
docker pull vllm/vllm-openai-rocm:latest
```

### 2.3 Verificación de instalación

```bash
$ vllm --version
vLLM version: 0.6.4
$ python -c "import vllm; print(vllm.__version__)"
0.6.4
$ python -c "import torch; print(torch.cuda.is_available())"
True
```

### 2.4 Versiones compatibles

| vLLM | CUDA | Python | Notas |
|------|------|--------|-------|
| 0.6.x | 12.1+ | 3.9-3.12 | Estable, recomendado |
| 0.7.x | 12.4+ | 3.10-3.12 | Más nuevo, mejor FP8 |
| main | 12.4+ | 3.10-3.12 | Última, inestable |

---

## 3. Elegir un modelo

### 3.1 Criterios de selección

| Criterio | Opciones |
|----------|----------|
| Tamaño (VRAM) | 0.5B → 1.76T (DeepSeek-V3) |
| Licencia | Apache 2.0, MIT, Llama Community, Gemma Terms |
| Tipo | Texto, visión, audio, multimodal mixto |
| Calidad | Benchmarks: MMLU, MMLU-Pro, GSM8K, HumanEval, MT-Bench |
| Velocidad | Tokens/s, latencia TTFT |
| Quantization nativa | Algunos modelos incluyen variantes GPTQ/AWQ/FP8 |

### 3.2 Modelos populares para empezar

| Modelo | Tamaño | VRAM FP16 | Licencia | Caso de uso |
|--------|-------:|----------:|----------|-------------|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | 2 GB | Llama | Tiny, edge |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 6 GB | Llama | Edge, demos |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | 14 GB | Apache | Default chat |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 14 GB | Apache | Default chat |
| `meta-llama/Mlama-3.1-8B-Instruct` | 8B | 16 GB | Llama | Calidad/precio |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | 28 GB | Apache | Mejor calidad |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | 140 GB | Llama | Producción |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | 144 GB | Apache | Top calidad |
| `deepseek-ai/DeepSeek-V3` | 671B (MoE) | 700+ GB | DeepSeek | Top open source |

> **Recomendación para empezar**: `Qwen/Qwen2.5-7B-Instruct` en una sola A100 o RTX 4090. Calidad excelente, Apache 2.0, razonablemente pequeño.

### 3.3 HuggingFace token

Algunos modelos (Llama, Gemma) requieren autenticación:

```bash
huggingface-cli login
# pega tu token de https://huggingface.co/settings/tokens
```

Y acepta la licencia en la página del modelo en HuggingFace.

---

## 4. El primer servidor

### 4.1 Comando mínimo

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0
```

Lo que pasa:
1. Descarga el modelo (~14 GB) en `~/.cache/huggingface/`.
2. Carga el modelo en GPU.
3. Compila los kernels CUDA.
4. Arranca el servidor HTTP en `:8000`.
5. Espera requests.

La primera ejecución tarda varios minutos (descarga + compilación). Las siguientes son instantáneas.

### 4.2 Validación con curl

```bash
# Health check
curl http://localhost:8000/v1/models
```

Output esperado:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-7B-Instruct",
      "object": "model",
      "created": 1700000000,
      "owned_by": "vllm"
    }
  ]
}
```

Probemos una completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Explica PagedAttention en una frase."}
    ]
  }'
```

Respuesta típica:

```json
{
  "id": "cmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "PagedAttention es una técnica de gestión de memoria..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 35,
    "completion_tokens": 87,
    "total_tokens": 122
  }
}
```

### 4.3 Streaming con curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Cuenta hasta 5."}],
    "stream": true
  }' --no-buffer
```

Verás tokens llegar uno a uno como `data: {...}\n\n` (formato SSE).

### 4.4 Consume con el SDK de OpenAI

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM no requiere key por default
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "Eres un asistente conciso."},
        {"role": "user", "content": "¿Qué es vLLM?"}
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)
```

> **Por qué `api_key="EMPTY"`**: la API de vLLM es OpenAI-compatible pero no implementa auth por default. Cualquier string no-vacía es aceptada. Para producción, configura auth (ver [[08 - Observabilidad y Deployment|módulo 08]]).

### 4.5 Streaming con el SDK

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Recita un haiku sobre GPU."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## 5. Flags de servidor esenciales

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 8192 \         # contexto máximo (prompt + completion)
  --gpu-memory-utilization 0.9 \ # fracción de VRAM para el modelo + KV cache
  --dtype bfloat16 \             # fp16, bf16, auto
  --max-num-seqs 256 \           # máximo de sequences concurrentes
  --max-num-batched-tokens 2048 \ # tokens totales por iteración
  --enable-chunked-prefill \     # estabilidad de latencia
  --served-model-name qwen-7b \  # alias para el API
  --api-key secret123            # requiere auth
```

| Flag | Default | Qué hace |
|------|---------|----------|
| `--port` | 8000 | Puerto del servidor HTTP |
| `--host` | localhost | Interfaz de red |
| `--max-model-len` | del modelo | Contexto total máximo (suma prompt + completion) |
| `--gpu-memory-utilization` | 0.9 | Fracción de VRAM a usar |
| `--dtype` | auto | fp16 / bf16 / fp32 |
| `--max-num-seqs` | 256 | Sequences concurrentes máximo |
| `--max-num-batched-tokens` | 2048 | Tokens por iteración (afecta throughput) |
| `--enable-chunked-prefill` | false | Estabilidad de latencia |
| `--served-model-name` | repo | Nombre público del modelo |
| `--api-key` | - | Activa auth (Bearer token) |
| `--tensor-parallel-size` | 1 | GPUs para tensor parallelism |
| `--pipeline-parallel-size` | 1 | GPUs para pipeline parallelism |
| `--quantization` | None | awq / gptq / bitsandbytes / fp8 |
| `--enable-prefix-caching` | false | Cache de prefijos compartidos |
| `--enable-chunked-prefill` | false | Prefill dividido |
| `--disable-log-requests` | false | Silencia logs de requests |

---

## 6. Logs y diagnóstico

### 6.1 Qué buscar en los logs

```
INFO 12-23 10:15:23 model_runner.py:1018] Loading model weights took 4.23 GB
INFO 12-23 10:15:25 worker.py:251] Memory usage: 12.4 GB (model: 14.0 GB, KV cache: 1.5 GB)
INFO 12-23 10:15:25 scheduler.py:187] Adding requests: 50
INFO 12-23 10:15:26 metrics.py:464] Avg generation throughput: 1247.3 tokens/s
INFO 12-23 10:15:26 metrics.py:464] Avg prompt throughput: 892.1 tokens/s
```

### 6.2 Probar con un benchmark sintético

```bash
vllm bench serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 128 \
  --num-prompts 100 \
  --request-rate 10
```

Output: latencias, throughput, TTFT, TPOT, errores. Es la forma estándar de validar antes de producción.

### 6.3 Monitoreo en vivo

```bash
# 1) GPU
watch -n 1 nvidia-smi

# 2) Logs estructurados
vllm serve ... --log-level info  # debug, info, warning, error

# 3) Métricas Prometheus en :8000/metrics
curl http://localhost:8000/metrics
```

---

## 7. Estructura de archivos recomendada

```
proyecto-vllm/
├── .env                       # HF_TOKEN, API_KEY, etc.
├── docker-compose.yml         # Para producción
├── Dockerfile                 # Imagen custom si necesitas deps extra
├── scripts/
│   ├── start.sh               # Wrapper con flags
│   ├── bench.sh               # Benchmark
│   └── load_test.py           # Locust/wrk
├── config/
│   └── vllm.yaml              # Config completa (opcional)
├── tests/
│   ├── test_api.py            # Test de endpoints
│   └── fixtures.json          # Prompts de prueba
└── logs/
    └── vllm.log
```

### 7.1 `start.sh` reutilizable

```bash
#!/usr/bin/env bash
set -euo pipefail

source .env

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEM:-0.9}
MAX_LEN=${MAX_LEN:-8192}

vllm serve "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --api-key "$VLLM_API_KEY" \
  --served-model-name "${MODEL_ALIAS:-$(basename $MODEL)}" \
  2>&1 | tee -a logs/vllm.log
```

---

## 8. Errores comunes de instalación

| Error | Causa | Solución |
|-------|-------|----------|
| `CUDA version mismatch` | `torch` instalado manualmente | `pip install vllm` solo, deja que instale torch |
| `OSError: libcudart.so` | CUDA libs no encontradas | Reinstala con `pip install --force-reinstall vllm` |
| `OutOfMemory al cargar` | VRAM insuficiente | Baja `--gpu-memory-utilization` o usa cuantización |
| `Permission denied: /dev/nvidia*` | Container sin GPUs | `--gpus all` en docker, `privileged` si es rootless |
| `OSError: model not found` | Repo mal escrito o sin auth | Verifica `huggingface-cli login` y nombre del repo |
| `Address already in use` | Puerto ocupado | Cambia `--port` o mata el proceso |

---

## 9. Smoke test: script end-to-end

```python
#!/usr/bin/env python
"""Smoke test del servidor vLLM: health, completion, streaming, embeddings."""
import time
from openai import OpenAI


def main():
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # 1) Health: listar modelos
    print("=== /v1/models ===")
    models = client.models.list()
    for m in models.data:
        print(f"  {m.id}")

    # 2) Chat completion bloqueante
    print("\n=== chat.completions (bloqueante) ===")
    t0 = time.time()
    resp = client.chat.completions.create(
        model=models.data[0].id,
        messages=[{"role": "user", "content": "Di 'OK'."}],
        max_tokens=10,
    )
    print(f"  Latency: {(time.time() - t0) * 1000:.0f} ms")
    print(f"  Response: {resp.choices[0].message.content!r}")
    print(f"  Usage: {resp.usage}")

    # 3) Streaming
    print("\n=== chat.completions (streaming) ===")
    t0 = time.time()
    first_token_t = None
    full = []
    stream = client.chat.completions.create(
        model=models.data[0].id,
        messages=[{"role": "user", "content": "Cuenta hasta 3."}],
        max_tokens=20,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_t is None:
                first_token_t = time.time()
            full.append(chunk.choices[0].delta.content)
    print(f"  TTFT: {(first_token_t - t0) * 1000:.0f} ms")
    print(f"  Total: {(time.time() - t0) * 1000:.0f} ms")
    print(f"  Response: {''.join(full)!r}")


if __name__ == "__main__":
    main()
```

Output esperado de un servidor sano:

```
=== /v1/models ===
  Qwen/Qwen2.5-7B-Instruct

=== chat.completions (bloqueante) ===
  Latency: 287 ms
  Response: 'OK.'
  Usage: CompletionUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16)

=== chat.completions (streaming) ===
  TTFT: 78 ms
  Total: 412 ms
  Response: '1, 2, 3.'
```

💡 **Siguiente paso**: en [[03 - API OpenAI-Compat|el siguiente módulo]] exploramos a fondo los endpoints: chat, completions legacy, embeddings, function calling, structured outputs y sampling parameters. Es donde vLLM pasa de "servidor que responde" a "motor de producción configurable".
