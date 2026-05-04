# 🧮 Introducción a Redes Neuronales

Las redes neuronales artificiales son el pilar del Deep Learning moderno. Su relevancia radica en la capacidad de aproximar funciones arbitrarias mediante composiciones no lineales de transformaciones simples, permitiendo aprender representaciones jerárquicas directamente desde los datos sin necesidad de ingeniería de características manual.

En Machine Learning, esto significa que un modelo puede capturar patrones complejos —desde bordes en imágenes hasta relaciones sintácticas en texto— si disponemos de suficientes datos y capacidad computacional.

---

## 1. El Perceptrón: La Unidad Fundamental

Un perceptrón recibe un vector de entrada $\mathbf{x} \in \mathbb{R}^n$, aplica una combinación lineal ponderada por un vector de pesos $\mathbf{w} \in \mathbb{R}^n$ y un sesgo $b \in \mathbb{R}$, y produce una salida a través de una función de activación $f$:

$$
z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
$$

$$
\hat{y} = f(z)
$$

Caso real: en los sistemas de detección de fraudes bancarios, un perceptrón puede actuar como clasificador binario inicial que pondera variables como monto, frecuencia y ubicación para emitir una probabilidad de fraude.

⚠️ **Advertencia:** Un perceptrón sin función de activación no lineal es equivalente a una regresión lineal, sin importar cuántas capas apiles. La no linealidad es indispensable para modelar fronteras de decisión complejas.

---

## 2. Funciones de Activación

La activación introduce la no linealidad que permite a la red aprender mapeos complejos.

### 2.1 ReLU (Rectified Linear Unit)

$$
f(z) = \max(0, z)
$$

Su derivada es trivial:

$$
f'(z) = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z \leq 0 \end{cases}
$$

Caso real: ReLU es el estándar de facto en CNNs modernas (ResNet, EfficientNet) porque acelera la convergencia y evita el problema del gradiente desvanecido en la región positiva.

### 2.2 Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Derivada:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

### 2.3 Tanh

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

Derivada:

$$
\tanh'(z) = 1 - \tanh^2(z)
$$

### 2.4 Softmax

Usada en la capa de salida para clasificación multiclase, convierte logits en distribuciones de probabilidad:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

| Activación | Rango | ¿Dónde usarla? | Problema conocido |
|------------|-------|----------------|-------------------|
| ReLU | $[0, \infty)$ | Capas ocultas | Neuronas muertas (salida siempre 0) |
| Sigmoid | $(0, 1)$ | Salida binaria | Gradiente desvanecido en extremos |
| Tanh | $(-1, 1)$ | Capas ocultas (RNNs clásicas) | Gradiente desvanecido en extremos |
| Softmax | $(0, 1)$ | Salida multiclase | Inestabilidad numérica con logits grandes |

💡 **Tip:** Usa `torch.nn.functional.relu` o `torch.relu`. Para evitar neuronas muertas, considera LeakyReLU: $f(z) = \max(\alpha z, z)$ con $\alpha$ pequeño (ej. 0.01).

---

## 3. Forward Pass

El forward pass es la evaluación secuencial de la red desde la entrada hasta la salida. Para una red de $L$ capas:

$$
\mathbf{a}^{[0]} = \mathbf{x}
$$

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})
$$

La salida final $\hat{y} = \mathbf{a}^{[L]}$ se compara con la etiqueta real $y$ mediante la función de pérdida.

---

## 4. Funciones de Pérdida

La pérdida cuantifica el error del modelo.

### 4.1 Error Cuadrático Medio (MSE)

Para regresión:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

### 4.2 Cross-Entropy

Para clasificación, mide la divergencia entre la distribución verdadera (one-hot) y la predicha:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

Caso real: en sistemas de recomendación de contenido (Netflix, Spotify), la cross-entropy categoriza interacciones del usuario entre miles de ítems.

💡 **Tip mnemotécnico:** *"Cross-entropy castiga la confianza mal puesta"*. Si el modelo predice 0.99 para la clase correcta, la pérdida es casi nula; si predice 0.01, explota logarítmicamente.

---

## 5. Backpropagation y la Regla de la Cadena

Backpropagation no es más que la aplicación sistemática de la regla de la cadena del cálculo diferencial para calcular $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ y $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$.

Para la capa $l$:

$$
\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} \odot f'(\mathbf{z}^{[l]})
$$

Luego, propagando hacia atrás:

$$
\delta^{[l-1]} = \left( (\mathbf{W}^{[l]})^\top \delta^{[l]} \right) \odot f'(\mathbf{z}^{[l-1]})
$$

Los gradientes respecto a los parámetros son:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^\top
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}
$$

Este mecanismo es la razón por la que frameworks como PyTorch construyen un grafo computacional dinámico: cada operación registra su operación inversa para aplicar la regla de la cadena automáticamente.

⚠️ **Advertencia:** Inicializar todos los pesos a cero provoca que todas las neuronas de una capa aprendan exactamente lo mismo (simetría). Inicializa con distribuciones como Xavier (Glort) o Kaiming (He) según la activación.

---

## 6. Gradient Descent y Variantes

El objetivo es minimizar $\mathcal{L}(\theta)$ donde $\theta$ agrupa todos los parámetros.

### 6.1 Batch Gradient Descent

Usa todo el dataset:

$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t)
$$

### 6.2 Mini-batch Gradient Descent

Usa un subconjunto $\mathcal{B}$ de tamaño $B$:

$$
\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_{\theta} \mathcal{L}_i(\theta_t)
$$

### 6.3 Stochastic Gradient Descent (SGD)

Caso extremo de mini-batch con $B=1$. Introduce ruido útil que ayuda a escapar de mínimos locales locales.

| Método | Estabilidad | Velocidad | Memoria | Uso recomendado |
|--------|-------------|-----------|---------|-----------------|
| Batch | Alta | Lenta | Alta | Datasets pequeños |
| Mini-batch | Media | Rápida | Media | **Estándar en Deep Learning** |
| SGD | Baja | Muy rápida | Baja | Datasets masivos, online learning |

Caso real: el entrenamiento de GPT-4 utiliza variantes de mini-batch con tamaños enormes (millones de tokens) distribuidos en clusters, balanceando entre estabilidad del gradiente y throughput computacional.

💡 **Tip:** El learning rate $\eta$ es el hiperparámetro más crítico. Si es muy grande, la pérdida diverge; si es muy pequeña, el entrenamiento es impracticablemente lento. Una buena heurística inicial para Adam es $10^{-3}$; para SGD, $10^{-2}$ a $10^{-1}$.

---

## 7. Implementación en PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Definición de una red neuronal simple
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hiperparámetros
input_dim = 784
hidden_dim = 256
num_classes = 10
learning_rate = 0.001
num_epochs = 5

model = SimpleNet(input_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Bucle de entrenamiento simplificado
for epoch in range(num_epochs):
    # Aquí iría la carga de datos (DataLoader)
    # inputs, labels = ...
    # outputs = model(inputs)
    # loss = criterion(outputs, labels)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    pass
```

⚠️ **Advertencia:** Es un error común olvidar `optimizer.zero_grad()`. PyTorch acumula gradientes por defecto; si no los reinicias, los gradientes se sumarán a los de la iteración anterior y el entrenamiento fallará.

---

## 📦 Código de Compresión

```python
"""
Script completo que resume una red neuronal fully-connected
con forward pass, backpropagation y entrenamiento en PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# Datos sintéticos de ejemplo
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = NeuralNetwork(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_model(model, dataloader, criterion, optimizer, epochs=5)
```

---

## 🎯 Proyecto: Clasificador de Dígitos MNIST con Red Fully-Connected

**Descripción:**
Implementar una red neuronal densamente conectada para clasificar imágenes de dígitos manuscritos del dataset MNIST (28x28 píxeles, 10 clases). El objetivo es comprender el flujo completo de datos desde la imagen aplanada hasta la predicción de clase.

**Requisitos funcionales:**
1. Cargar MNIST usando `torchvision.datasets` y aplicar transformaciones para convertir imágenes a tensores y normalizar pixeles al rango $[-1, 1]$.
2. Definir una red con al menos dos capas ocultas con activación ReLU.
3. Utilizar CrossEntropyLoss como función de objetivo.
4. Entrenar con SGD durante al menos 10 epochs, registrando pérdida y accuracy por epoch.
5. Evaluar en el conjunto de test reportando accuracy global.
6. Visualizar 5 ejemplos mal clasificados junto con la probabilidad predicha.

**Componentes principales:**
- `MNISTDataset` y `DataLoader` con batch size de 64.
- Módulo `FCClassifier` con capas lineales y dropout del 20% para regularización.
- Script de entrenamiento con bucle de validación al final de cada epoch.
- Utilidad de logging con `tqdm` o métricas simples impresas.

**Métricas de éxito:**
- Accuracy en test $\geq 97\%$ con red fully-connected.
- Tiempo de entrenamiento por epoch menor a 30 segundos en CPU moderna.
- Pérdida de entrenamiento decreciente monotónicamente sin divergencia.

**Referencias:**
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Capítulos 6 y 8.
- PyTorch Documentation: `torch.nn`, `torch.optim`.
- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
