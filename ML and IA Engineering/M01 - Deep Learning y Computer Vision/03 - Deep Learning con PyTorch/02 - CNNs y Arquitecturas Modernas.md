# 🖼️ CNNs y Arquitecturas Modernas

Las Convolutional Neural Networks (CNNs) revolucionaron el campo de la visión por computadora al introducir la *invariancia traduccional* y el *compartimiento de pesos*. En lugar de tratar cada píxel de forma independiente, una CNN explota la estructura espacial local de las imágenes aprendiendo filtros que detectan patrones como bordes, texturas y formas en diferentes jerarquías.

En Machine Learning, esto significa que podemos entrenar modelos profundos con menos parámetros que una red fully-connected equivalente, logrando mejor generalización en tareas visuales.

---

## 1. Convolución 2D

La operación de convolución aplica un filtro (kernel) local sobre la imagen de entrada. Para una entrada $\mathbf{X}$ de tamaño $H \times W \times C_{in}$ y un kernel $\mathbf{K}$ de tamaño $K_h \times K_w \times C_{in} \times C_{out}$, la salida en la posición $(i, j)$ para el canal $c$ es:

$$
(\mathbf{X} * \mathbf{K})_{i,j,c} = \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} \sum_{d=0}^{C_{in}-1} \mathbf{X}_{i+m, j+n, d} \cdot \mathbf{K}_{m,n,d,c}
$$

El kernel se desliza (stride) sobre la imagen, computando productos punto locales. Este mecanismo es la base del *receptive field*: cada neurona en una capa convolucional solo ve una región local de la entrada, pero capas posteriores combinan estas visiones locales para formar comprensiones globales.

Caso real: los sistemas de conducción autónoma (Tesla, Waymo) utilizan CNNs para detectar vehículos, peatones y señales de tráfico en tiempo real a partir de cámaras frontales.

---

## 2. Padding y Stride

### 2.1 Padding

El padding agrega píxeles (típicamente ceros) alrededor del borde de la imagen para controlar el tamaño espacial de la salida y preservar información de los bordes.

Con padding $P$:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K_h}{S} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + 2P - K_w}{S} \right\rfloor + 1
$$

Donde $S$ es el stride.

### 2.2 Stride

El stride controla el desplazamiento del kernel. Un stride mayor reduce la dimensionalidad espacial más agresivamente, funcionando como un mecanismo de *downsampling* implícito.

| Configuración | Padding | Stride | Efecto típico |
|---------------|---------|--------|---------------|
| Same | $P = \frac{K-1}{2}$ | 1 | Mantiene tamaño espacial |
| Valid | $P = 0$ | 1 | Reduce tamaño espacial |
| Downsampling | $P = 0$ o 1 | 2 | Reduce a la mitad |

💡 **Tip:** En PyTorch, usa `padding='same'` (disponible en versiones recientes) o calcula manualmente. Para un kernel $3\times3$ con stride 1, usa padding 1 para preservar dimensiones.

---

## 3. Pooling

El pooling reduce la dimensionalidad espacial y proporciona una ligera invariancia a pequeñas traslaciones.

### Max Pooling

$$
y_{i,j} = \max_{m,n \in \text{ventana}} x_{i+m, j+n}
$$

### Average Pooling

$$
y_{i,j} = \frac{1}{K_h K_w} \sum_{m,n} x_{i+m, j+n}
$$

Caso real: en sistemas de reconocimiento facial (FaceID, Clearview AI), el max pooling ayuda a que la red sea robusta a ligeras variaciones en la posición de los rasgos faciales.

⚠️ **Advertencia:** El pooling excesivo puede causar pérdida de información espacial fina. En arquitecturas modernas como ResNet, el stride en capas convolucionales ha reemplazado parcialmente al pooling como mecanismo de reducción.

---

## 4. Feature Maps

Cada kernel produce un *feature map* (mapa de características). En capas tempranas, los filtros aprenden detectores de bordes y colores; en capas intermedias, texturas y patrones; en capas profundas, partes de objetos y configuraciones semánticas.

Visualizar los feature maps es una herramienta diagnóstica poderosa: si los mapas de una capa temprana están en blanco, probablemente la red está sufriendo de *dying ReLU* o el learning rate es demasiado alto.

---

## 5. Arquitecturas Clásicas

### 5.1 LeNet (1998)

Pionera en CNNs. Estructura: Conv -> Pool -> Conv -> Pool -> FC. Demostró que las convoluciones eran viables para reconocimiento de caracteres.

### 5.2 AlexNet (2012)

Ganó ImageNet 2012 por un margen enorme. Introdujo ReLU, dropout y GPUs. Arquitectura: 5 capas conv + 3 FC.

### 5.3 VGG (2014)

Filosofía de simplicidad: stacks de convoluciones $3\times3$ con padding same. Demostró que la profundidad es crucial. VGG-16 y VGG-19 son estándares.

### 5.4 ResNet (2015)

Introdujo las *skip connections* (conexiones residuales) para mitigar el degradado del gradiente en redes muy profundas (50, 101, 152 capas):

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

Donde $\mathcal{F}$ es el mapeo residual. Si la identidad es óptima, la red solo necesita aprender a producir ceros, lo cual es más fácil que aprender una identidad a través de una pila de capas no lineales.

### 5.5 DenseNet (2017)

Conecta cada capa con todas las capas posteriores dentro de un *dense block*. Promueve el reuso de características y reduce el número de parámetros:

$$
\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])
$$

| Arquitectura | Profundidad típica | Innovación clave | Parámetros (aprox) |
|--------------|--------------------|------------------|--------------------|
| LeNet | 8 | Primera CNN práctica | 60K |
| AlexNet | 8 | ReLU + Dropout + GPU | 60M |
| VGG-16 | 16 | Bloques $3\times3$ uniformes | 138M |
| ResNet-50 | 50 | Skip connections + bottleneck | 25.6M |
| DenseNet-121 | 121 | Conexiones densas | 8M |

Caso real: los modelos de detección de tumores en mamografías a menudo parten de ResNet-50 o DenseNet-121 pre-entrenados en ImageNet, aprovechando que los filtros de bajo nivel (bordes, texturas) son transferibles al dominio médico.

💡 **Tip mnemotécnico:** *"ResNet salta; DenseNet abraza"*. ResNet suma la entrada (skip connection), DenseNet concatena todas las entradas anteriores.

---

## 6. Batch Normalization en CNNs

La batch normalization normaliza las activaciones por canal dentro de un mini-batch:

$$
\hat{x}_{i} = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

En CNNs, se aplica por canal: si la salida de una capa conv tiene forma $(B, C, H, W)$, BN calcula $\mu$ y $\sigma$ sobre $(B, H, W)$ para cada canal $C$.

Beneficios:
- Permite learning rates más altos.
- Reduce la sensibilidad a la inicialización.
- Actúa como regularizador leve debido al ruido del mini-batch.

⚠️ **Advertencia:** En inferencia, BN usa estadísticas de población (media y varianza móviles), no las del mini-batch actual. PyTorch maneja esto automáticamente con `model.train()` y `model.eval()`.

---

## 7. Implementación en PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

---

## 📦 Código de Compresión

```python
"""
Script completo de CNN con arquitectura VGG-like,
BatchNorm, Dropout y entrenamiento en PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class VGGStyleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGStyleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders()
model = VGGStyleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    loss = train_epoch(model, train_loader, criterion, optimizer, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")
```

---

## 🎯 Proyecto: Clasificador de Objetos CIFAR-10 con ResNet-18 desde Cero

**Descripción:**
Construir una implementación desde cero de una arquitectura ResNet-18 (o una versión reducida ResNet-8) para clasificar imágenes a color del dataset CIFAR-10. El objetivo es internalizar el mecanismo de skip connections y el diseño de bloques residuales.

**Requisitos funcionales:**
1. Implementar un bloque residual básico (`BasicBlock`) con dos convoluciones $3\times3$, BatchNorm y una skip connection que sume la entrada a la salida.
2. Construir la red con 4 etapas (layers) donde cada etapa duplica los canales y reduce la mitad la resolución espacial mediante stride 2 en el primer bloque.
3. Aplicar data augmentation: random crop, random horizontal flip y normalización.
4. Entrenar con SGD + Momentum (0.9) y learning rate scheduling (ReduceLROnPlateau o StepLR).
5. Reportar accuracy top-1 en test y curvas de pérdida.
6. Implementar una función que visualice los feature maps de la primera capa convolucional para 3 imágenes de ejemplo.

**Componentes principales:**
- `resnet_block(in_channels, out_channels, stride)` con manejo de la proyección (`downsample`) cuando los canales o la resolución cambian.
- `ResNetCIFAR(num_classes=10)` con capa inicial, 4 layers de bloques, global average pooling y FC final.
- `train.py` con bucle de entrenamiento/validación, checkpointing y early stopping manual.

**Métricas de éxito:**
- Accuracy en test $\geq 85\%$ para CIFAR-10 con ResNet-18.
- Convergencia estable sin explosión de pérdida (indicador de skip connections funcionando).
- Tiempo de entrenamiento razonable (< 30 min en GPU modesta para 50 epochs).

**Referencias:**
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Huang, G., et al. (2017). Densely Connected Convolutional Networks. CVPR.
- PyTorch `torchvision.models` como referencia de implementación oficial.
