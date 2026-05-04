# 🚀 Transfer Learning

Entrenar una red neuronal profunda desde cero requiere cantidades masivas de datos y tiempo computacional. El transfer learning (aprendizaje por transferencia) resuelve esto aprovechando el conocimiento adquirido por un modelo pre-entrenado en un dataset grande y general (como ImageNet) para una tarea específica y usualmente más pequeña. Es una de las técnicas más impactantes para poner Deep Learning en producción con recursos limitados.

En ML práctico, el transfer learning reduce drásticamente el tiempo de entrenamiento, mejora la generalización en datasets pequeños y permite construir prototipos funcionales en horas en lugar de días.

---

## 1. Feature Extraction vs Fine-tuning

Existen dos paradigmas principales:

### 1.1 Feature Extraction

Congelamos todas las capas del modelo pre-entrenado y entrenamos únicamente un clasificador (típicamente una o dos capas fully-connected) al final.

- **Ventaja:** Entrenamiento rápido, riesgo mínimo de sobreajuste.
- **Desventaja:** Si la tarea destino difiere significativamente del dominio origen, los features extraídos pueden no ser óptimos.

### 1.2 Fine-tuning

Descongelamos (total o parcialmente) los pesos del modelo pre-entrenado y los actualizamos con un learning rate pequeño.

- **Ventaja:** El modelo se adapta a las particularidades del nuevo dominio.
- **Desventaja:** Mayor riesgo de sobreajuste y requiere más datos y tiempo.

| Estrategia | Datos necesarios | Tiempo de entrenamiento | Riesgo de sobreajuste | Cuándo usarla |
|------------|------------------|-------------------------|----------------------|---------------|
| Feature Extraction | Muy pocos | Minutos | Muy bajo | Dataset < 1,000 imágenes, dominio similar |
| Fine-tuning completo | Moderados | Horas | Alto | Dataset grande, dominio diferente |
| Fine-tuning progresivo | Moderados | Horas | Medio | Dataset mediano, busca balance |

Caso real: en la detección de neumonía en radiografías de tórax, los investigadores parten de ResNet-50 pre-entrenado en ImageNet. Las texturas de bajo nivel (bordes, contrastes) son universales, por lo que feature extraction ya da buenos resultados; sin embargo, fine-tuning de las últimas capas mejora la sensibilidad al adaptar los filtros a patrones pulmonares específicos.

---

## 2. Congelar y Descongelar Capas

En PyTorch, congelar pesos se hace desactivando el cálculo de gradientes:

```python
for param in model.parameters():
    param.requires_grad = False
```

Luego, al instanciar el optimizer, solo se pasan los parámetros entrenables:

```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

### Descongelado progresivo

Una estrategia avanzada es descongelar capas de forma gradual, empezando por las capas superiores (más específicas) y bajando hacia las inferiores (más generales):

1. Entrenar solo el clasificador final (epochs 1-5).
2. Descongelar las últimas dos capas convolucionales y entrenar con LR muy bajo (epochs 6-10).
3. Descongelar todo con LR aún menor (epochs 11-20).

💡 **Tip:** Las capas tempranas de una CNN aprenden características de bajo nivel (bordes, colores) que son casi universales. Las capas profundas aprenden patrones de alto nivel (formas de objetos) que son específicos del dataset original. Por eso, en dominios similares, solo se fine-tunean las capas finales.

---

## 3. Learning Rate Diferenciado

No todas las capas deberían actualizarse con la mismo learning rate. Las capas pre-entrenadas necesitan LR más pequeños porque ya contienen pesos útiles; el clasificador nuevo necesita LR mayor porque parte de valores aleatorios.

En PyTorch, se configura con *parameter groups*:

```python
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},      # Clasificador nuevo
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Capas profundas
    {'params': model.layer1.parameters(), 'lr': 1e-5}   # Capas tempranas
])
```

O usa discriminative fine-tuning, donde cada grupo de capas tiene un LR escalado por un factor (ej. 0.1 por grupo anterior).

⚠️ **Advertencia:** Si aplicas el mismo learning rate agresivo a todo el modelo pre-entrenado, puedes destruir rápidamente los features útiles aprendidos en ImageNet. Siempre usa LR más bajos para capas inferiores.

---

## 4. Domain Adaptation

Cuando el dominio destino difiere drásticamente del origen (ej. fotos naturales -> imágenes de satélite o médicas), el transfer learning simple puede no ser suficiente. El domain adaptation busca reducir la discrepancia entre distribuciones de origen y destino.

Técnicas comunes:
- **Fine-tuning masivo:** descongelar todo el modelo.
- **Adversarial domain adaptation:** entrenar un discriminador que no distinga entre dominios mientras el extractor de features aprende representaciones invariantes.
- **Pseudo-labeling:** usar el modelo en el dominio destino para generar etiquetas confidentes y re-entrenar.

Caso real: en agricultura de precisión, los modelos pre-entrenados en ImageNet se transfieren a imágenes de dron en infrarrojo cercano (NDVI). Los colores son completamente diferentes, por lo que se requiere fine-tuning completo o técnicas de domain adaptation adversarial.

---

## 5. Modelos Pre-entrenados en PyTorch

`torchvision.models` proporciona decenas de modelos con pesos pre-entrenados en ImageNet:

```python
from torchvision import models

# Cargar ResNet-50 pre-entrenado
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Reemplazar la última capa para clasificación binaria
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Congelar capas base
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo la nueva capa
for param in model.fc.parameters():
    param.requires_grad = True
```

Modelos populares disponibles:
- **ResNet-18/34/50/101/152**
- **EfficientNet-B0 a B7**
- **MobileNet-V2/V3** (para edge/mobile)
- **Vision Transformer (ViT)**
- **RegNet, ConvNeXt**

| Modelo | Parámetros | Top-1 ImageNet | Ideal para |
|--------|------------|----------------|------------|
| ResNet-50 | 25.6M | 76.1% | Uso general, balance tamaño/rendimiento |
| EfficientNet-B0 | 5.3M | 77.1% | Dispositivos con recursos limitados |
| EfficientNet-B4 | 19M | 83.4% | Alta accuracy, costo computacional moderado |
| ViT-B/16 | 86M | 81.1% | Datasets masivos, capacidad de generalización alta |

---

## 6. Implementación en PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

class TransferClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(TransferClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Ejemplo de fine-tuning con learning rates diferenciados
model = TransferClassifier(num_classes=10, freeze_backbone=False)

# Grupos de parámetros con LR diferenciado
optimizer = optim.Adam([
    {'params': model.backbone.layer1.parameters(), 'lr': 1e-5},
    {'params': model.backbone.layer2.parameters(), 'lr': 1e-5},
    {'params': model.backbone.layer3.parameters(), 'lr': 1e-4},
    {'params': model.backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': model.backbone.fc.parameters(), 'lr': 1e-3}
])
```

---

## 📦 Código de Compresión

```python
"""
Script completo de Transfer Learning con PyTorch.
Incluye carga de ResNet-50 pre-entrenado, congelación,
reemplazo de capa final, fine-tuning progresivo y
learning rate diferenciado.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

class TransferResNet(nn.Module):
    def __init__(self, num_classes, unfreeze_from_layer=None):
        super(TransferResNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Congelar todo por defecto
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar capas a partir de un punto
        if unfreeze_from_layer is not None:
            for name, param in self.backbone.named_parameters():
                if unfreeze_from_layer in name:
                    param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # Usar CIFAR-10 como proxy de dataset pequeño
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(test_ds, batch_size=64)

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
train_loader, test_loader = get_data_loaders()

# Fase 1: Feature extraction
model = TransferResNet(num_classes=10, unfreeze_from_layer=None).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.backbone.fc.parameters(), lr=1e-3)

print("=== Fase 1: Feature Extraction ===")
for epoch in range(3):
    loss = train_epoch(model, train_loader, criterion, optimizer, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# Fase 2: Fine-tuning de últimas capas
for param in model.backbone.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': model.backbone.fc.parameters(), 'lr': 1e-3}
])

print("\n=== Fase 2: Fine-Tuning ===")
for epoch in range(3):
    loss = train_epoch(model, train_loader, criterion, optimizer, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")
```

---

## 🎯 Proyecto: Clasificador de Especies de Plantas con Transfer Learning

**Descripción:**
Desarrollar un sistema de clasificación multiclase para identificar especies de plantas a partir de fotografías de hojas. El dataset es pequeño (~200 imágenes por clase, 38 clases), lo que lo hace ideal para transfer learning. El objetivo es comparar tres estrategias: feature extraction pura, fine-tuning parcial y fine-tuning completo, midiendo cuál generaliza mejor.

**Requisitos funcionales:**
1. Cargar un modelo EfficientNet-B0 pre-entrenado en ImageNet desde `torchvision.models`.
2. Implementar tres variantes del modelo:
   a. `feature_extractor`: congela todo el backbone.
   b. `fine_tune_partial`: descongela las últimas dos etapas (blocks 6 y 7).
   c. `fine_tune_full`: descongela todo.
3. Aplicar data augmentation apropiada para fotos de plantas: rotación, volteo, cambios de brillo, recorte aleatorio.
4. Usar learning rate diferenciado: LR más alto para el clasificador nuevo, más bajo para capas pre-entrenadas.
5. Entrenar cada variante durante 20 epochs con early stopping (patience=5).
6. Evaluar y comparar accuracy, macro-F1 y matrices de confusión entre las tres variantes.
7. Generar un informe que indique cuál estrategia fue óptima y por qué.

**Componentes principales:**
- `models.py`: definición de las tres variantes de EfficientNet-B0 con switches de congelación.
- `data_module.py`: `PlantDataset` con transforms diferenciados para train y val.
- `trainer.py`: loop de entrenamiento con soporte para múltiples grupos de parámetros en el optimizer.
- `experiment.py`: script que ejecuta los tres experimentos secuencialmente y guarda resultados.
- `report.py`: generación de gráficos comparativos de curvas de entrenamiento.

**Métricas de éxito:**
- La variante de fine-tuning parcial debe superar en al menos 5 puntos porcentuales a la de feature extraction pura.
- Fine-tuning completo no debe sobreajustar en más de 10% de diferencia entre train y val accuracy.
- Tiempo de entrenamiento total de los tres experimentos < 2 horas en GPU modesta.

**Referencias:**
- Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NIPS.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- PyTorch Docs: `torchvision.models`, `torch.optim` parameter groups.
