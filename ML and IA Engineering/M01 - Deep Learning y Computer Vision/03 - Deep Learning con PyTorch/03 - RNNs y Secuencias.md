# 🔄 RNNs y Secuencias

Los datos secuenciales —texto, audio, series temporales, genomas— no pueden modelarse adecuadamente con redes feed-forward porque carecen de memoria del contexto previo. Las Recurrent Neural Networks (RNNs) abordan esta limitación manteniendo un *estado oculto* que actúa como memoria de las entradas pasadas, permitiendo modelar dependencias temporales.

En Machine Learning, dominar las RNNs es esencial para tareas como traducción automática, generación de texto, predicción financiera y análisis de sentimiento.

---

## 1. Motivación para Secuencias

A diferencia de una imagen, donde todos los píxeles están disponibles simultáneamente, una secuencia $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ tiene una dimensión temporal. La predicción de $x_t$ puede depender de $x_{t-1}, x_{t-2}, \ldots$.

Un modelo de lenguaje, por ejemplo, estima:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

Caso real: los asistentes de voz (Siri, Alexa) utilizan arquitecturas secuenciales para transcribir audio a texto y luego interpretar el comando como una secuencia de palabras.

---

## 2. RNN Simple (Vanilla RNN)

En cada paso de tiempo $t$, la RNN actualiza su estado oculto $\mathbf{h}_t$ y emite una salida $\mathbf{y}_t$:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

Donde $\mathbf{W}_{hh}$ es la matriz de pesos recurrentes (la misma en todos los pasos de tiempo, lo que constituye el *compartimiento de pesos* temporal).

Este diseño hace que la red sea conceptualmente muy profunda en el tiempo: desenrollar una secuencia de longitud $T$ equivale a una red feed-forward con $T$ capas que comparten parámetros.

⚠️ **Advertencia:** La RNN simple es difícil de entrenar en secuencias largas debido al problema del gradiente desvanecido (y, en menor medida, exploding gradient). Los gradientes que fluyen hacia atrás en el tiempo se multiplican repetidamente por $\mathbf{W}_{hh}^\top$, lo que hace que decaigan o exploten exponencialmente.

💡 **Tip:** Si tu problema tiene secuencias cortas (< 20 pasos), una vanilla RNN puede funcionar. Para secuencias largas, usa LSTM o GRU sin dudarlo.

---

## 3. Long Short-Term Memory (LSTM)

LSTM fue diseñada explícitamente para mitigar el gradiente desvanecido mediante una *celda de memoria* $\mathbf{c}_t$ y tres compuertas que regulan el flujo de información.

### 3.1 Ecuaciones de LSTM

**Forget gate:** decide qué información de la celda anterior se descarta.

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**Input gate:** decide qué nueva información se almacena.

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
$$

**Actualización de celda:**

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

**Output gate:** decide qué parte de la celda se expone como estado oculto.

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

La clave está en la actualización de la celda: es una suma ponderada, no una multiplicación encadenada. Esto crea un *camino de derivadas constante* a través del tiempo, permitiendo que los gradientes fluyan sin atenuación.

Caso real: Google Translate (antes de Transformer) utilizaba secuencias de LSTM bidireccionales con atención para modelar dependencias a largo plazo entre idiomas con orden de palabras diferente.

---

## 4. GRU (Gated Recurrent Unit)

GRU simplifica LSTM fusionando la celda de memoria y el estado oculto, y utilizando solo dos compuertas:

**Update gate:**

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

**Reset gate:**

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

**Candidato de estado:**

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])
$$

**Nuevo estado:**

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

| Aspecto | LSTM | GRU |
|---------|------|-----|
| Compuertas | 3 (input, forget, output) | 2 (update, reset) |
| Estados | Celda + oculto | Solo oculto |
| Parámetros | Más | Menos (~25% menos) |
| Rendimiento | Ligeramente superior en tareas largas | Comparable, más rápido de entrenar |

💡 **Tip mnemotécnico:** *"LSTM tiene memoria separada (celda); GRU la mezcla todo en uno"*. Si tienes recursos limitados, empieza con GRU.

---

## 5. Bidireccionalidad

En muchas tareas (NER, reconocimiento de voz), el contexto futuro es tan importante como el pasado. Una RNN bidireccional apila dos RNNs: una procesa la secuencia de izquierda a derecha ($\overrightarrow{\mathbf{h}}_t$) y otra de derecha a izquierda ($\overleftarrow{\mathbf{h}}_t$). La salida final concatena ambos estados:

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]
$$

⚠️ **Advertencia:** Las bidireccionales no son aplicables a tareas de predicción en tiempo real donde el futuro no está disponible (ej. trading en vivo), pero son ideales para análisis offline de texto o audio.

Caso real: los modelos de reconocimiento de entidades nombradas (NER) en spaCy utilizan LSTMs bidireccionales para capturar que en la frase *"Apple lanzó un nuevo iPhone"*, la palabra *Apple* es una organización gracias al contexto posterior.

---

## 6. Secuencia a Secuencia (Seq2Seq)

El modelo seq2seq consta de dos partes:
- **Encoder:** una RNN que comprime la secuencia de entrada en un vector de contexto $\mathbf{c}$ (típicamente el último estado oculto).
- **Decoder:** una RNN que genera la secuencia de salida palabra por palabra condicionada a $\mathbf{c}$.

$$
\mathbf{c} = \text{Encoder}(x_1, \ldots, x_T)
$$

$$
s_t = \text{Decoder}(s_{t-1}, y_{t-1}, \mathbf{c})
$$

---

## 7. Teacher Forcing

Durante el entrenamiento del decoder, en lugar de alimentar la predicción anterior $\hat{y}_{t-1}$ (que puede ser errónea y desviar al modelo), se alimenta la etiqueta real $y_{t-1}$. Esto acelera la convergencia y estabiliza el entrenamiento.

En inferencia, obviamente, no disponemos de $y_{t-1}$ real, por lo que se usa $\hat{y}_{t-1}$ (o técnicas como beam search).

⚠️ **Advertencia:** El teacher forcing puro puede causar *exposure bias*: el modelo nunca ve sus propios errores durante el entrenamiento. Técnicas como *scheduled sampling* mitigan esto usando predicciones propias con probabilidad creciente durante el entrenamiento.

---

## 8. Implementación en PyTorch

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Usar el último estado oculto concatenado si es bidireccional
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        out = self.fc(hidden)
        return out
```

---

## 📦 Código de Compresión

```python
"""
Script completo de LSTM para clasificación de secuencias.
Incluye embedding, LSTM bidireccional y bucle de entrenamiento.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h = torch.cat((h_forward, h_backward), dim=1)
        h = self.dropout(h)
        out = self.fc(h)
        return out

# Datos sintéticos
vocab_size = 1000
seq_length = 20
num_samples = 500
sequences = [torch.randint(1, vocab_size, (seq_length,)).tolist() for _ in range(num_samples)]
labels = [torch.randint(0, 3, (1,)).item() for _ in range(num_samples)]

dataset = TextDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BiLSTMClassifier(vocab_size, embed_dim=128, hidden_dim=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

---

## 🎯 Proyecto: Análisis de Sentimiento en Reseñas de Productos

**Descripción:**
Construir un clasificador de sentimiento (positivo/neutro/negativo) sobre reseñas de productos utilizando una arquitectura LSTM bidireccional. El proyecto abarca desde el preprocesamiento de texto hasta la evaluación con métricas de clasificación multiclase.

**Requisitos funcionales:**
1. Tokenizar y construir un vocabulario a partir del dataset de reseñas, limitando el vocabulario a las 10,000 palabras más frecuentes.
2. Implementar padding/truncado para que todas las secuencias tengan longitud fija (ej. 200 tokens).
3. Definir un modelo `SentimentBiLSTM` con capa de embedding pre-entrenable, 2 capas LSTM bidireccionales, dropout (0.5) y capa lineal final.
4. Entrenar con CrossEntropyLoss y Adam, usando early stopping basado en validation loss.
5. Evaluar con accuracy, precision, recall y F1-score por clase.
6. Generar una matriz de confusión y analizar ejemplos mal clasificados.

**Componentes principales:**
- `vocab_builder.py`: construcción de vocabulario y mapeo token -> índice.
- `dataset.py`: `Dataset` de PyTorch que gestiona secuencias y etiquetas.
- `model.py`: arquitectura BiLSTM con manejo de padding.
- `train.py`: bucle de entrenamiento con logging de métricas y guardado de checkpoints.
- `evaluate.py`: cálculo de métricas y visualización de resultados.

**Métricas de éxito:**
- Macro-F1 en validación $\geq 0.75$.
- Diferencia entre accuracy de entrenamiento y validación menor a 8% (indicador de buena generalización).
- Convergencia estable en menos de 20 epochs.

**Referencias:**
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. EMNLP.
- PyTorch Docs: `torch.nn.LSTM`, `torch.nn.utils.rnn.pad_sequence`.
