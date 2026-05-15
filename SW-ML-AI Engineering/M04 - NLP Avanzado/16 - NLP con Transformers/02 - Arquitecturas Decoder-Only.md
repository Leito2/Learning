# 🚀 02 - Arquitecturas Decoder-Only

Los modelos decoder-only representan el paradigma dominante en la era de los **Large Language Models (LLMs)**. A diferencia de los encoders, que consumen toda la secuencia simultáneamente, los decoders generan texto de manera **autoregresiva**: predicen el siguiente token condicionado únicamente en los tokens anteriores. Esta restricción, lejos de ser una limitación, les otorga una flexibilidad generativa sin precedentes y ha demostrado propiedades emergentes de razonamiento y aprendizaje in-contexto a escala masiva.

---

## 1. Causal Language Modeling (CLM)

La base matemática del decoder-only es la **máscara causal** (triangular inferior). Dada una secuencia $x = (x_1, x_2, \ldots, x_T)$, el modelo factoriza la probabilidad conjunta como un producto de probabilidades condicionales:

$$
P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t}; \theta)
$$

En el mecanismo de atención, esto se implementa reemplazando los valores del triángulo superior de la matriz de scores por $-\infty$ antes del softmax:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Donde $M_{ij} = 0$ si $j \leq i$ (posición anterior o actual) y $M_{ij} = -\infty$ si $j > i$ (posiciones futuras).

```mermaid
graph TD
    A[Input: "The cat sat"] --> B[Embedding + Positional]
    B --> C[Masked Self-Attention]
    C --> D[Feed-Forward]
    D --> E[Output: "on"]
    E --> F[Next Step]
    F --> C
    style C fill:#f9f,stroke:#333,stroke-width:2px
```

⚠️ **Advertencia**: La máscara causal es **no negociable** en decoders puros. Eliminarla convierte al modelo en un encoder, rompiendo la propiedad autoregresiva y causando data leakage durante el entrenamiento.

---

## 2. La Familia GPT

### 2.1 GPT-1: El Paradigma de Preentrenamiento Generalista

GPT-1 (Radford et al., 2018) estableció el principio fundamental: **preentrenar en un corpus masivo sin supervisión, luego fine-tunar en tareas específicas**.

- **Arquitectura**: Decoder de 12 capas, 768 dim, 12 heads, 117M parámetros.
- **Pretraining**: BooksCorpus (7,000 libros sin publicar).
- **Fine-tuning**: Añadir capa lineal sobre el estado oculto del último token.

### 2.2 GPT-2: Escalamiento y Zero-Shot

GPT-2 escaló a 1.5B parámetros y demostró que un modelo suficientemente grande podía realizar tareas sin fine-tuning explícito, simplemente condicionando el prompt adecuadamente.

**Diferencias clave con GPT-1**:
- LayerNorm movido a la entrada de cada sub-bloque (pre-norm vs post-norm).
- Inicialización de pesos escalada por $\frac{1}{\sqrt{N}}$ donde $N$ es el número de capas.
- Vocabulario ampliado a 50,257 tokens (BPE en bytes).
- Context window de 1024 tokens (vs 512 de GPT-1).

### 2.3 GPT-3: Emergencia del Few-Shot Learning

GPT-3 (Brown et al., 2020) escaló drásticamente la frontera hasta 175B parámetros, revelando capacidades emergentes:

| Modelo | Parámetros | Capas | $d_{model}$ | Batch Size | Context |
|--------|------------|-------|-------------|------------|---------|
| GPT-3 Small | 125M | 12 | 768 | 0.5M | 2048 |
| GPT-3 Medium | 350M | 24 | 1024 | 0.5M | 2048 |
| GPT-3 Large | 760M | 24 | 1536 | 0.5M | 2048 |
| GPT-3 XL | 1.3B | 24 | 2048 | 1M | 2048 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 1M | 2048 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 2M | 2048 |
| GPT-3 13B | 13B | 40 | 5140 | 2M | 2048 |
| GPT-3 175B | 175B | 96 | 12288 | 3.2M | 2048 |

La ecuación de scaling laws de Kaplan et al. predice la pérdida $L$ en función de parámetros $N$ y datos $D$:

$$
L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

Donde $\alpha_N \approx 0.073$ y $\alpha_D \approx 0.095$ para GPT-3.

Caso real: GitHub Copilot se basa en una variante de Codex (descendiente de GPT-3) fine-tuned en código. Demostró que los decoders autoregresivos capturan patrones sintácticos y semánticos de lenguajes de programación con extraordinaria fidelidad.

---

## 3. Context Window Scaling

Una limitación histórica de los decoders es el tamaño fijo del contexto (tipicamente 1024-2048 tokens en GPT-3). Técnicas modernas extienden esto:

- **Alibi / Rotary Embeddings (RoPE)**: Codifican posiciones de forma relativa, permitiendo extrapolación a longitudes mayores que las vistas en entrenamiento.
- **Sparse Attention**: Patterns como local + global (Longformer), sliding window (BigBird) o dilated attention reducen complejidad de $O(n^2)$ a $O(n \log n)$ o $O(n)$.
- **Positional Interpolation**: Escalar linealmente los índices posicionales para adaptar un modelo entrenado en 2048 a 32768 tokens.

💡 **Tip**: La atención local windowed es especialmente efectiva en código y documentos técnicos, donde la relevancia suele estar concentrada en vecindarios cercanos.

---

## 4. In-Context Learning Emergente

GPT-3 popularizó tres modalidades de aprendizaje sin actualizar parámetros:

1. **Zero-shot**: El prompt describe la tarea directamente.
2. **One-shot**: Se incluye un ejemplo de entrada-salida antes de la consulta.
3. **Few-shot**: Se incluyen $k$ ejemplos (típicamente $k=5$ a $64$).

La capacidad de ICL emerge bruscamente al cruzar ciertos umbrales de escala (~10B-100B parámetros). La hipótesis teórica sugiere que los modelos grandes actúan como **meta-optimizadores implícitos**: durante el forward pass, las activaciones se ajustan dinámicamente según los ejemplos del prompt.

$$
P(y \mid x, C) = \text{softmax}\left(W \cdot f_\theta(x, C)\right)
$$

Donde $C = \{(x_1, y_1), \ldots, (x_k, y_k)\}$ es el contexto de ejemplos y $f_\theta$ es el modelo.

---

## 5. GPT-4 y Más Allá: ¿Qué Cambia Arquitecturalmente?

OpenAI no ha revelado detalles completos de GPT-4, pero se sabe que:
- Es **multimodal** (acepta texto + imágenes).
- Utiliza una arquitectura de **mixture of experts (MoE)** estimada, donde solo un subconjunto de parámetros se activa por token, permitiendo escalar a ~1.8T parámetros totales con costo de inferencia similar a un modelo denso de ~200B.
- Emplea técnicas avanzadas de alineamiento RLHF (Reinforcement Learning from Human Feedback).

| Aspecto | GPT-3 (175B) | GPT-4 (estimado) |
|---------|-------------|------------------|
| Parámetros activos/token | ~175B | ~220B (de ~1.8T total) |
| Contexto máximo | 2K / 4K | 8K / 32K / 128K |
| Modalidad | Solo texto | Texto + Imagen |
| Arquitectura | Densa | MoE (especulado) |
| Entrenamiento | Predicción siguiente token | RLHF + verificación |

---

## 6. Decoder-Only para Clasificación

Aunque los decoders están optimizados para generación, pueden adaptarse a tareas de comprensión. Dado que **no existe token `[CLS]`**, las estrategias alternativas incluyen:

1. **Último token oculto**: Usar el estado del último token de la secuencia.
2. **Mean pooling**: Promediar todos los estados ocultos de la secuencia.
3. **Token especial insertado**: Añadir un token `<|endoftext|>` al final y usar su representación.
4. **Logit-based**: Concatenar la secuencia con cada label posible y seleccionar la que maximice la probabilidad logarítmica (zero-shot classification).

```python
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn

class GPT2ForClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
        mean_pooled = sum_embeddings / mask_expanded.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(mean_pooled)
        return logits
```

⚠️ **Advertencia**: Para clasificación, los encoders (BERT/RoBERTa) suelen superar a los decoders del mismo tamaño en datasets pequeños, dado que la bidireccionalidad captura mejor representaciones discriminativas. Los decoders brillan cuando se escalan masivamente o cuando la tarea es inherentemente generativa.

---

## 📦 Código de Compresión

Generación autoregresiva manual sin usar `.generate()`:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "In the field of machine learning, transformers"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

model.eval()
with torch.no_grad():
    for _ in range(20):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        # Greedy decoding
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(generated)
```

---

## 🎯 Proyecto Documentado: Generador de Documentación Técnica

**Objetivo**: Fine-tunear GPT-2 (o usar few-shot con GPT-3.5 API) para generar docstrings y documentación a partir de código Python.

**Dataset**: 10,000 pares (código, documentación) de repositorios open-source.

**Estrategia**:
1. Formatear cada ejemplo como: `CODE: {code}\nDOC: {doc}\n###`
2. Fine-tuning con CLM por 3 epochs, LR $5e-5$.
3. Evaluación con BLEU y revisión humana de coherencia técnica.
4. Comparar few-shot (sin fine-tuning) vs fine-tuned en tareas de dominio específico.

Ver integración con sistemas de QA en [[05 - Caso Practico - Sistema de Question Answering]].
