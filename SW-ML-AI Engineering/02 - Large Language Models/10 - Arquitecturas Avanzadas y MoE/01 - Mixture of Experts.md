# 🧩 Mixture of Experts (MoE)

Las arquitecturas densas, donde todos los parámetros participan en el cálculo de cada token, han alcanzado límites prácticos de escalabilidad. Entrenar un modelo de 500B parámetros en precisión FP16 requiere más de 1 TB de memoria GPU, un recurso accesible solo para un puñado de organizaciones. Las redes **Mixture of Experts (MoE)** ofrecen una vía de escape: escalar el número total de parámetros para aumentar la capacidad del modelo, manteniendo el costo computacional por token cercano al de un modelo mucho más pequeño. En esta nota, exploramos los fundamentos matemáticos, los algoritmos de enrutamiento y los sistemas MoE que ya dominan la vanguardia de la IA.

---

## 1. Sparsely Activated Networks

En una red densa tradicional, la salida de una capa feed-forward (FFN) se computa como:

$$
y = \sigma(W x + b)
$$

donde $W \in \mathbb{R}^{d_{out} \times d_{in}}$ es una matriz densa. En un modelo de 7B parámetros con dimensiones $d=4096$, una capa FFN típica tiene $d_{ff} = 11008$, resultando en ~45M parámetros por capa y ~32 capas.

Una red **sparsely activated** divide esta capa FFN en $N$ "expertos", cada uno siendo una FFN más pequeña $E_i(x) = \sigma(W_i x + b_i)$. Sin embargo, en lugar de sumar la contribución de todos los expertos, un mecanismo de enrutamiento selecciona un subconjunto $K \ll N$ de expertos para cada token:

$$
y = \sum_{i \in \mathcal{K}} G(x)_i \cdot E_i(x)
$$

Si $N=8$ y $K=2$, solo el 25% de los parámetros FFN se activan por token. El modelo total puede tener 8 veces más parámetros FFN, pero el costo computacional por token es equivalente a un modelo ~2x más grande, no 8x.

$$
\text{Costo}_{MoE} = O(K \cdot d_{ff} \cdot d) \ll O(N \cdot d_{ff} \cdot d)
$$

Caso real: **Google's GLaM** (Du et al., 2021) utiliza 64 expertos por capa con un total de 1.2 trillones de parámetros, pero solo activa 97B por token, logrando una eficiencia comparable a un modelo denso 7x más pequeño en entrenamiento.

---

## 2. Gating Network

La **red de enrutamiento** (gating network) es el corazón de un sistema MoE. Decide qué expertos procesan cada token. Dado un token de entrada $x \in \mathbb{R}^d$, la red de enrutamiento produce un vector de scores:

$$
G(x) = \text{Softmax}(W_g \cdot x + b_g)
$$

donde $W_g \in \mathbb{R}^{N \times d}$. El score $G(x)_i$ representa la probabilidad de que el experto $i$ sea el más adecuado para el token $x$.

### Top-k Gating

El método más común es **Top-k routing**: seleccionar los $k$ expertos con mayor score y renormalizar sus probabilidades:

$$
\mathcal{K} = \text{TopK}(G(x), k), \quad G'(x)_i = \frac{G(x)_i}{\sum_{j \in \mathcal{K}} G(x)_j} \cdot \mathbb{1}_{i \in \mathcal{K}}
$$

La salida de la capa MoE es entonces:

$$
y = \sum_{i \in \mathcal{K}} G'(x)_i \cdot E_i(x)
$$

Para $k=1$ (Switch Transformer), la fórmula se simplifica a $y = E_{i^*}(x)$ donde $i^* = \arg\max_i G(x)_i$.

Caso real: **Mixtral 8x7B** utiliza $k=2$, seleccionando 2 de sus 8 expertos por token. Esto significa que, aunque el modelo tiene 46.7B parámetros FFN en total, solo ~13B están activos en cualquier forward pass.

💡 **Tip:** La inicialización de $W_g$ es crítica. Una inicialización demasiado grande puede causar que todos los tokens se enruten a un solo experto al inicio del entrenamiento, provocando colapso del routing. Se recomienda inicializar con desviaciones pequeñas (ej. $\sigma = 10^{-2}$).

---

## 3. Load Balancing Loss

Un problema fundamental del enrutamiento es la tendencia al **colapso de expertos**: la red de enrutamiento puede aprender a enviar la mayoría de los tokens a unos pocos expertos "comodines", dejando otros expertos sin entrenar. Esto reduce la capacidad efectiva del modelo y genera cuellos de botella en hardware distribuido.

Para mitigarlo, se introduce una **pérdida de balanceo de carga** (load balancing loss). Sea $f_i$ la fracción de tokens enrutados al experto $i$ en un batch, y $P_i$ la media de las probabilidades de enrutamiento asignadas al experto $i$:

$$
f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}_{[\text{token } t \text{ va a } i]}, \quad P_i = \frac{1}{T} \sum_{t=1}^{T} G(x_t)_i
$$

La loss de balanceo es:

$$
\mathcal{L}_{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

Donde $\alpha$ es un hiperparámetro de regularización (típicamente $10^{-2}$). Esta loss penaliza la correlación entre la fracción de tokens asignados y la confianza promedio del router: si un experto recibe muchos tokens y tiene alta probabilidad asignada, la loss es alta. El mínimo global se alcanza cuando $f_i = P_i = \frac{1}{N}$ para todo $i$.

⚠️ **Advertencia:** Un $\alpha$ demasiado alto fuerza un balanceo mecánico perfecto, lo cual puede ser subóptimo si ciertos tokens del dominio (ej. código, matemáticas) genuinamente requieren más capacidad que otros. Se recomienda ajustar $\alpha$ con validación en un held-out set.

---

## 4. Switch Transformer

El **Switch Transformer** (Fedus et al., 2022) simplifica el MoE tradicional reemplazando el Top-k con **Top-1 routing**: cada token se asigna exactamente a un experto. La capa se define como:

$$
y = E_{\arg\max_i G(x)_i}(x)
$$

Esto reduce la sobrecarga de comunicación en sistemas distribuidos, ya que cada token solo necesita enviarse a una GPU/TPU destino (asumiendo que los expertos están distribuidos). La loss total de entrenamiento incluye la cross-entropy del modelo, la load balancing loss y, opcionalmente, una **router z-loss** que penaliza logits del router excesivamente grandes para mejorar la estabilidad numérica:

$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{balance} + \beta \mathcal{L}_{z}
$$

con:

$$
\mathcal{L}_{z} = \frac{1}{T} \sum_{t=1}^{T} \left( \log \sum_{i=1}^{N} e^{z_{t,i}} \right)^2
$$

Caso real: Google entrenó Switch Transformers con hasta 1.6 trillones de parámetros, demostrando que los modelos MoE escalaban de forma predecible y superaban a los modelos densos en eficiencia de entrenamiento (pre-training loss vs FLOPs).

---

## 5. Mixtral 8x7B

**Mixtral 8x7B** (Jiang et al., 2024) es la demostración práctica más exitosa de MoE en el ecosistema open-source. Sus especificaciones clave son:

| Característica | Valor |
|----------------|-------|
| Expertos totales | 8 |
| Parámetros por experto | ~7B |
| Parámetros totales | ~46.7B |
| Expertos activos por token | 2 |
| Parámetros activos por token | ~13B |
| Contexto máximo | 32k tokens |

En benchmarks como MMLU, HellaSwag y GSM8K, Mixtral 8x7B iguala o supera a LLaMA-2-70B, a pesar de requerir aproximadamente la mitad de memoria en inferencia y un throughput significativamente mayor gracias a su menor carga computacional por token.

La clave de su éxito radica en que la red de enrutamiento aprende a especializar expertos de forma semántica: expertos dedicados a matemáticas, a código, a conversación casual, etc. Este fenómeno de especialización emergente fue observado empíricamente analizando qué tokens activaban cada experto.

---

## 6. Routing Algorithms

Más allá del Top-k básico, se han propuesto variantes sofisticadas:

### Noisy Top-k Gating

Añade ruido gaussiano a los logits del router antes del Top-k para fomentar la exploración durante el entrenamiento:

$$
G(x) = \text{TopK}(\text{Softmax}(W_g x + b_g + \epsilon), k), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

### Expert Choice Routing

En lugar de que el router asigne tokens a expertos, los expertos "eligen" qué tokens procesar. Cada experto selecciona los $k'$ tokens con mayor afinidad, permitiendo un balanceo perfecto de carga (cada experto procesa exactamente $k'$ tokens). El desafío es que algunos tokens pueden no ser seleccionados por ningún experto o ser seleccionados por varios.

| Algoritmo | Balanceo | Complejidad | Uso Típico |
|-----------|----------|-------------|------------|
| Top-1 (Switch) | Medio | Baja | Entrenamiento masivo |
| Top-k | Variable | Media | Modelos generales |
| Noisy Top-k | Mejorado | Media | Evita colapso temprano |
| Expert Choice | Perfecto | Alta | Cargas heterogéneas |

---

## 7. Capacity Factor

El **capacity factor** es un hiperparámetro crítico en sistemas MoE distribuidos. Define cuántos tokens puede procesar cada experto por batch. Si la capacidad de un experto se fija en $C$ tokens, y el batch tiene $B$ tokens, el capacity factor es:

$$
\text{Capacity Factor} = \frac{C}{B/N} = \frac{C \cdot N}{B}
$$

Un factor de 1.0 significa que cada experto tiene espacio exacto para su cuota promedio de tokens. En la práctica, se utiliza un factor > 1 (ej. 1.25 o 1.5) para manejar la varianza en la asignación de tokens. Si un experto recibe más tokens que su capacidad, los tokens excedentes se descartan (*dropped tokens*), lo que introduce una pérdida de información.

$$
\text{Dropped Tokens} = \sum_{i=1}^{N} \max(0, T_i - C)
$$

Caso real: En el entrenamiento de Switch Transformer, Google utilizó un capacity factor de 1.0 para maximizar la eficiencia del TPU, aceptando una pequeña tasa de dropped tokens (~0.5%) que no impactó significativamente la convergencia.

⚠️ **Advertencia:** Un capacity factor muy alto desperdicia memoria y computación. Uno muy bajo degrada la calidad del modelo. Se recomienda ajustar este valor mediante un grid search en una fracción del dataset de pre-entrenamiento.

---

## 8. Comparativa Dense vs Sparse

| Propiedad | Modelo Denso | Modelo MoE |
|-----------|--------------|------------|
| Parámetros totales | $P$ | $N \cdot P_{expert}$ |
| Parámetros activos | $P$ | $K \cdot P_{expert}$ |
| Memoria en inferencia | $2 \cdot P$ bytes | $2 \cdot P_{active}$ + overhead routing |
| Throughput (tokens/s) | Base | Mayor (menos FLOPs por token) |
| Calidad downstream | Base | Superior a igual FLOPs de entrenamiento |
| Entrenamiento distribuido | Simple | Complejo (all-to-all communication) |
| Fine-tuning | Directo | Requiere técnicas especiales (ej. Expert Weights Averaging) |

Caso real: **Microsoft** utiliza variantes de MoE en sus modelos de traducción de Bing, logrando una mejora de 15% en BLEU sobre modelos densos de igual latencia de inferencia.

---

## 📦 Código de Compresión: Capa MoE en PyTorch

El siguiente script implementa una capa MoE con Top-k routing, load balancing loss y capacidad de broadcasting para entrenamiento en GPU única.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, num_experts=8, top_k=2, balance_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_coef = balance_coef
        
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        bsz, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*S, D]
        
        # Routing
        gate_logits = self.gate(x_flat)  # [B*S, N]
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (topk_indices == i).any(dim=-1)  # [B*S]
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                # Weight by corresponding gate prob
                weights = gate_probs[mask, i:i+1]
                output[mask] += weights * expert_output
        
        # Load balancing loss
        router_prob_per_expert = gate_probs.mean(dim=0)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (topk_indices == i).float().sum()
        tokens_per_expert = tokens_per_expert / (bsz * seq_len * self.top_k)
        
        balance_loss = self.num_experts * (router_prob_per_expert * tokens_per_expert).sum()
        
        return output.view(bsz, seq_len, d_model), self.balance_coef * balance_loss

# Demo
moe = MoELayer(d_model=512, d_ff=2048, num_experts=8, top_k=2)
x = torch.randn(2, 16, 512)
out, aux_loss = moe(x)
print(f"Salida: {out.shape}, Aux Loss: {aux_loss.item():.4f}")
```

---

## 🎯 Proyecto: Entrenamiento de un Modelo MoE Toy

Diseña un modelo de lenguaje de 2 capas transformer donde la segunda capa FFN se reemplace por una capa MoE con 4 expertos y top_k=1 (Switch). Entrena el modelo en una tarea de clasificación de secuencias (ej. sentiment analysis en IMDB o una tarea sintética de predicción de tokens).

### Métricas de Evaluación

| Métrica | Modelo Denso | Modelo MoE |
|---------|--------------|------------|
| Accuracy | Baseline | Objetivo: +2% |
| FLOPs por forward | $F$ | ~$0.5F$ |
| Distribución de expertos | - | Entropía > 0.8 |

El proyecto debe incluir:

- Visualización de la utilización de cada experto a lo largo del entrenimiento.
- Gráfica de la load balancing loss vs. cross-entropy.
- Análisis de si ciertos tipos de tokens (puntuación, verbos, nombres propios) tienden a activar expertos específicos.

💡 **Tip:** Utiliza `torch.nn.Embedding` para asignar IDs a tokens y analiza la correlación entre ID de token y experto activado. Puedes visualizar esto con un heatmap de seaborn.

⚠️ **Advertencia:** El entrenamiento de MoE requiere precisión numérica estable. Usa siempre `torch.float32` para los logits del router y los cálculos de softmax, incluso si el resto del modelo usa mixed precision (FP16/BF16).

---

![Arquitectura de red neuronal](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/640px-Colored_neural_network.svg.png)

---

**Enlaces internos:**
- [[00 - Bienvenida]]
- [[02 - State Space Models (Mamba)]]
- [[03 - Modelos Multimodales de LLMs]]
- [[04 - Caso Practico - Agente Autonomo con LLM]]
