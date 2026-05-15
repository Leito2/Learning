# 🔧 04 - Fine-Tuning para NLP Tasks

El preentrenamiento masivo en corpus no etiquetados dota a los Transformers de representaciones lingüísticas universales. Sin embargo, el verdadero valor se desbloquea en el **fine-tuning**: adaptar estos modelos generales a tareas específicas mediante entrenamiento supervisado en datasets de dominio. Este proceso de transfer learning es la razón por la cual un modelo entrenado en Wikipedia puede clasificar sentimientos, identificar entidades o responder preguntas con precisión cercana al estado del arte.

---

## 1. Clasificación de Texto

### 1.1 Fundamento Teórico

Dado un texto $x$, el modelo encoder (ej. BERT) produce un vector de contexto $h_{[CLS]} \in \mathbb{R}^d$ asociado al token especial `[CLS]`. Una capa lineal con softmax lo proyecta al espacio de clases:

$$
P(y \mid x) = \text{softmax}(W \cdot h_{[CLS]} + b)
$$

La función de pérdida es entropía cruzada:

$$
\mathcal{L}_{\text{CLS}} = - \sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{I}(y_i = c) \log P(y_i = c \mid x_i)
$$

### 1.2 Análisis de Sentimiento

Caso real: Las plataformas de e-commerce como Amazon utilizan modelos fine-tuneados sobre BERT/RoBERTa para clasificar automáticamente millones de reseñas diarias en categorías de satisfacción, detectando problemas de productos en tiempo real.

```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(5000)),
    eval_dataset=dataset["test"].shuffle(seed=42).select(range(1000)),
)

trainer.train()
```

⚠️ **Advertencia**: En clasificación binaria con datasets desbalanceados, monitorear accuracy puede ser engañoso. Usar **F1-macro** o **AUC-ROC** para una evaluación justa.

---

## 2. Named Entity Recognition (NER)

NER es una tarea de **clasificación a nivel de token**. Cada token $x_i$ se clasifica en una de las clases BIO (Beginning, Inside, Outside):

$$
P(y_i \mid x) = \text{softmax}(W \cdot h_i + b)
$$

Donde $h_i$ es el estado oculto del token $i$ en la última capa del encoder.

### 2.1 Esquema BIO

| Tag | Significado |
|-----|-------------|
| O | Fuera de entidad |
| B-PER | Inicio de persona |
| I-PER | Continuación de persona |
| B-ORG | Inicio de organización |
| I-ORG | Continuación de organización |
| B-LOC | Inicio de localización |
| I-LOC | Continuación de localización |

```python
from transformers import BertForTokenClassification, BertTokenizerFast
from datasets import load_dataset

dataset = load_dataset("conll2003")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

label_list = dataset["train"].features["ner_tags"].feature.names

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))
```

💡 **Tip**: Usar `BertTokenizerFast` (backend Rust) es crucial para alinear subword tokens (ej. "##ing") con las etiquetas originales a nivel de palabra. El valor `-100` es ignorado por `CrossEntropyLoss` en PyTorch.

Caso real: Sistemas de cumplimiento financiero en bancos utilizan NER fine-tuned para detectar y anonimizar automáticamente datos personales (PII) en documentos legales antes de compartirlos con terceros.

---

## 3. Question Answering (QA)

### 3.1 QA Extractivo

El modelo predice dos probabilidades sobre el contexto: inicio ($s$) y fin ($e$) de la respuesta.

$$
P(s \mid C, Q) = \text{softmax}(W_s \cdot H_C + b_s)
$$

$$
P(e \mid C, Q) = \text{softmax}(W_e \cdot H_C + b_e)
$$

Donde $H_C$ son los estados ocultos del contexto. La respuesta es el span que maximiza $P(s) \cdot P(e)$ sujeto a $s \leq e$.

```python
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris."

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores) + 1

answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
)
print(f"Answer: {answer}")
```

### 3.2 QA Generativo

Usando encoder-decoder (T5/BART), el modelo genera la respuesta libremente:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

input_text = "question: What is the capital of France? context: France is in Europe. Paris is its capital."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4. Summarization y Translation

Tareas seq2seq por excelencia. La métrica estándar es **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) para resumen, y **BLEU** (Bilingual Evaluation Understudy) para traducción.

$$
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}
$$

Donde $R_{lcs}$ y $P_{lcs}$ son recall y precision basadas en la subsecuencia común más larga.

---

## 5. Prompt-Based Fine-Tuning

En lugar de añadir capas de clasificación, el fine-tuning basado en prompts reformula la tarea como completado de texto. Dada una plantilla:

```
Review: {text}
Sentiment: {mask}
```

El modelo predice el token correspondiente a "positive" o "negative". Esto alinea mejor el preentrenamiento MLM/CLM con el fine-tuning.

💡 **Tip**: Prompt-based learning es especialmente útil con few-shot examples, donde un modelo grande (GPT-3/4) realiza la tarea sin gradient updates.

---

## 6. Adapter Layers para NLP

En lugar de fine-tunear todos los parámetros del modelo base, los **adapters** (Houlsby et al., 2019) insertan capas pequeñas entrenables dentro de cada bloque Transformer:

```
Adapter(x) = x + W_{up} \cdot \text{ReLU}(W_{down} \cdot x)
```

Donde $W_{down} \in \mathbb{R}^{d \times r}$ y $W_{up} \in \mathbb{R}^{r \times d}$ con $r \ll d$ (típicamente $r=16$ o $64$).

| Método | Parámetros entrenables | Rendimiento relativo |
|--------|------------------------|----------------------|
| Full Fine-tuning | 100% | 100% (baseline) |
| Adapter (r=64) | ~3-5% | ~97-99% |
| Adapter (r=16) | ~1% | ~95-97% |
| LoRA | ~0.1-1% | ~98-99% |

```python
from transformers import BertForSequenceClassification
from transformers.adapters import BertAdapterModel

# Con adapters, se entrena solo el adapter + classification head
model = BertAdapterModel.from_pretrained("bert-base-uncased")
model.add_adapter("sentiment")
model.train_adapter("sentiment")
model.add_classification_head("sentiment", num_labels=2)
```

⚠️ **Advertencia**: Aunque los adapters reducen memoria y tiempo de entrenamiento, para datasets muy grandes (>100k ejemplos), el full fine-tuning suele alcanzar el techo de rendimiento absoluto.

---

## 7. Multi-Task Learning

Entrenar un solo modelo en múltiples tareas simultáneamente puede mejorar la generalización mediante representaciones compartidas:

$$
\mathcal{L}_{\text{MTL}} = \sum_{t=1}^{T} \lambda_t \cdot \mathcal{L}_t
$$

Donde $\lambda_t$ son pesos de tarea. Estrategias comunes:
- **Uniforme**: $\lambda_t = 1/T$
- **Dinámico**: Escalar por magnitud inversa de la pérdida (GradNorm).
- **Uncertainty weighting**: Kendall et al. aprenden $\lambda_t$ como parámetros.

Caso real: Google utilizó un modelo multi-task T5 (MT5) entrenado simultáneamente en traducción, resumen y QA para mejorar la transferencia entre tareas en su asistente virtual.

---

## 8. Catastrophic Forgetting en NLP

Cuando se fine-tunea un modelo preentrenado en una tarea específica, puede perder capacidades generales aprendidas durante el preentrenamiento. Esto se conoce como **olvidado catastrófico**.

### 8.1 Mitigaciones

1. **Regularización por distancia**: Penalizar cambios grandes en los pesos originales:
   $$
   \mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{i} ||\theta_i - \theta_i^{0}||^2
   $$
   (Elastic Weight Consolidation - EWC).

2. **Replay**: Mezclar datos del corpus original durante el fine-tuning.
3. **Adapters/LoRA**: Congelar el modelo base y solo entrenar capas pequeñas.
4. **Continual Learning**: Entrenar secuencialmente en tareas nuevas preservando rendimiento en antiguas.

---

## 📦 Código de Compresión

Pipeline unificado con Hugging Face para múltiples tareas:

```python
from transformers import pipeline

# Clasificación
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("I love this product!"))

# NER
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Hugging Face is a company based in New York."))

# QA Extractivo
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
print(qa(question="Who founded Hugging Face?", context="Hugging Face was founded by French entrepreneurs."))

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(summarizer("Your long text here...", max_length=130, min_length=30))

# Translation
translator = pipeline("translation_en_to_de", model="t5-base")
print(translator("The transformer is a powerful architecture."))
```

---

## 🎯 Proyecto Documentado: Plataforma Multi-Task NLP

**Objetivo**: Construir una API que exponga múltiples endpoints (clasificación, NER, QA, summarization) usando un único backend basado en adapters.

**Arquitectura**:
1. Modelo base: `roberta-base` (congelado).
2. Adapter "sentiment" para clasificación.
3. Adapter "ner" para token classification.
4. Modelo T5 con LoRA para summarization y QA generativo.
5. FastAPI para servir predicciones con baja latencia.

**Evaluación**:
- Latencia P95 < 200ms por request.
- Comparar rendimiento full fine-tune vs adapters en cada tarea.
- Medir forgetting en GLUE después de entrenar cada adapter secuencialmente.

Ver aplicación concreta en [[05 - Caso Practico - Sistema de Question Answering]].
