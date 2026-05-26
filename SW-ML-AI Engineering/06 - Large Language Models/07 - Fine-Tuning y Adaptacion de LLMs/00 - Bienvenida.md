# 🤖 Fine-Tuning y Adaptación de LLMs: Índice y Fundamentos

La adaptación de Large Language Models (LLMs) representa uno de los pilares fundamentales de la ingeniería de ML moderna. Mientras que los modelos pre-entrenados capturan conocimiento general del lenguaje, su verdadero valor empresarial y científico emerge cuando son especializados en dominios específicos mediante técnicas de fine-tuning y adaptación parameter-efficient.

---

## 1. Estructura del Curso

Este curso está organizado en seis módulos que cubren desde los fundamentos del entrenamiento supervisado hasta la construcción de sistemas RAG productivos.

| Módulo | Título | Descripción |
|--------|--------|-------------|
| 00 | Bienvenida | Índice, glosario y objetivos del curso |
| 01 | [[01 - Fine-Tuning Supervisado\|Fine-Tuning Supervisado]] | Full fine-tuning, catastrophic forgetting y estrategias de optimización |
| 02 | [[02 - LoRA y PEFT\|LoRA y PEFT]] | Métodos parameter-efficient: LoRA, QLoRA, adapters y prompt tuning |
| 03 | [[03 - Prompt Engineering Avanzado\|Prompt Engineering Avanzado]] | CoT, ReAct, self-consistency y técnicas de prompting estructurado |
| 04 | [[04 - In-Context Learning y RAG\|In-Context Learning y RAG]] | Retrieval-Augmented Generation, dense retrieval y vector stores |
| 05 | [[05 - Caso Practico - Asistente Especializado con RAG\|Caso Práctico]] | Construcción end-to-end de un asistente médico/legal |

---

## 2. Glosario Técnico

A continuación se definen los términos esenciales que se utilizarán a lo largo del curso:

**Fine-Tuning Supervisado (SFT):** Proceso de actualizar los pesos de un modelo pre-entrenado $\theta$ mediante gradient descent sobre un dataset etiquetado $\mathcal{D} = \{(x_i, y_i)\}$, minimizando la cross-entropy:

$$\mathcal{L}_{\text{SFT}} = -\sum_{(x,y) \in \mathcal{D}} \sum_{t=1}^{|y|} \log P_\theta(y_t | y_{<t}, x)$$

**LoRA (Low-Rank Adaptation):** Técnica que reparametriza las matrices de peso mediante una descomposición de bajo rango:

$$W = W_0 + BA$$

donde $W_0 \in \mathbb{R}^{d \times k}$ son pesos congelados, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ y $r \ll \min(d,k)$.

**QLoRA (Quantized LoRA):** Extensión de LoRA que cuantiza $W_0$ a 4-bit (NormalFloat4), utiliza double quantization y paging optimizado para entrenar modelos de 65B parámetros en una sola GPU de 48GB.

**Prefix Tuning:** Método que añade vectores entrenables (prefixes) a las activaciones de cada capa del transformer, manteniendo todos los parámetros del modelo congelados.

**Prompt Tuning:** Similar al prefix tuning pero restringido a embeddings de input; aprende "soft prompts" de longitud fija.

**PEFT (Parameter-Efficient Fine-Tuning):** Familia de métodos que buscan adaptar LLMs entrenando $<1\%$ de los parámetros totales.

**Adapter:** Capas bottleneck inyectadas entre subcapas del transformer, entrenables mientras el cuerpo principal permanece congelado.

**Instruction Tuning:** Fine-tuning sobre datasets de instrucciones $(instruction, input, output)$ para alinear el modelo con tareas descritas en lenguaje natural.

**RLHF (Reinforcement Learning from Human Feedback):** Marco de entrenamiento por refuerzo que utiliza preferencias humanas para optimizar una política de lenguaje mediante PPO sobre un modelo de recompensa.

**RAG (Retrieval-Augmented Generation):** Arquitectura que combina un componente de recuperación $\mathcal{R}$ con un generador $\mathcal{G}$:

$$p(y|x) = \sum_{z \in \mathcal{R}(x)} p(z|x) p(y|x,z)$$

---

## 3. Objetivos de Aprendizaje

Al finalizar este curso, serás capaz de:

1. Diseñar pipelines de fine-tuning supervisado con mitigación de catastrophic forgetting.
2. Seleccionar y aplicar métodos PEFT (LoRA, QLoRA, Adapters) según restricciones de hardware y datos.
3. Construir prompts avanzados (CoT, ReAct) que maximicen el rendimiento de modelos cerrados y abiertos.
4. Implementar sistemas RAG end-to-end con dense retrieval, reranking y evaluación por métricas de fidelidad.
5. Desplegar un asistente especializado (médico/legal) con verificación de hechos y trazabilidad de fuentes.

---

Caso real: **OpenAI y ChatGPT**. El modelo GPT-3.5/4 base fue adaptado mediante un pipeline de tres fases: (1) SFT sobre ejemplos de diálogo etiquetados, (2) entrenamiento de un modelo de recompensa con comparaciones humanas, y (3) optimización RLHF con PPO. Este proceso demostró que la adaptación estratégica supera el simple escalado de datos pre-entrenados.

![Diagrama conceptual de adaptación de LLMs](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Artificial_intelligence_brain.svg/512px-Artificial_intelligence_brain.svg.png)

```mermaid
graph LR
    A[Pre-trained LLM] --> B[SFT]
    B --> C[RM Training]
    C --> D[RLHF / PPO]
    D --> E[Adapted Model]
    E --> F[PEFT / LoRA]
    F --> G[Domain Specialist]
```

⚠️ **Advertencia:** El fine-tuning no inyecta nuevo conocimiento factual fiable de manera directa; solo reestructura el conocimiento existente. Para datos actualizados o específicos, combina siempre con RAG.

💡 **Tip:** Antes de fine-tunar, evalúa si prompt engineering o RAG resuelven el problema. El fine-tuning debe ser la última opción cuando se requiere cambio de estilo, formato o comportamiento consistente.

---

## 🎯 Proyecto del Curso: Asistente Especializado con RAG

El proyecto final consiste en un asistente médico/legal que combine fine-tuning parameter-efficient con un sistema de recuperación documental. Se desarrollará en cinco fases paralelas a los módulos:

- **Fase 1:** Preparación del modelo base y estrategia de fine-tuning supervisado.
- **Fase 2:** Aplicación de LoRA/QLoRA para adaptación en hardware consumer.
- **Fase 3:** Diseño de prompts estructurados y chain-of-thought para razonamiento jurídico/clínico.
- **Fase 4:** Implementación de pipeline RAG con FAISS y reranking por cross-encoder.
- **Fase 5:** Evaluación con métricas de faithfulness, answer relevance y context precision.

[[01 - Fine-Tuning Supervisado]]
