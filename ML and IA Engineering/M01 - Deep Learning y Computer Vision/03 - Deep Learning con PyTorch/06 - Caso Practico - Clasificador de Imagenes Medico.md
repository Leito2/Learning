# 🏥 Caso Práctico: Clasificador de Imágenes Médico

La aplicación de Deep Learning en imágenes médicas es uno de los campos de mayor impacto social. Desde la detección de tumores hasta la clasificación de enfermedades degenerativas, los modelos de visión pueden asistir a los radiólogos reduciendo tiempos de diagnóstico y capturando patrones sutiles que el ojo humano puede pasar por alto.

Sin embargo, el dominio médico impone restricciones únicas: datasets pequeños y desbalanceados, alta dimensionalidad de imágenes (radiografías, tomografías), necesidad de interpretabilidad para ganar la confianza clínica, y métricas especializadas que van más allá de la simple accuracy.

---

## 1. Problemática del Clasificador Médico

Consideremos un clasificador binario que determina si una radiografía de tórax presenta neumonía. La formulación es:

$$
\hat{y} = f_{\theta}(\mathbf{X}), \quad \mathbf{X} \in \mathbb{R}^{H \times W \times C}
$$

Donde $\mathbf{X}$ es la imagen médica y $\hat{y} \in [0, 1]$ la probabilidad de patología.

Las particularidades del dominio médico incluyen:
- **Desbalanceo extremo:** la prevalencia de una enfermedad puede ser del 1-5% en datasets de screening.
- **Anonimización:** los datos deben cumplir regulaciones (HIPAA, GDPR).
- **Etiquetado costoso:** solo médicos especialistas pueden etiquetar correctamente.
- **Interpretabilidad obligatoria:** los clínicos necesitan saber *por qué* el modelo emitió un diagnóstico.

Caso real: el modelo CheXNet de Stanford alcanzó rendimiento de radiólogo en la detección de neumonía en radiografías de tórax utilizando DenseNet-121 pre-entrenado en ImageNet, demostrando que el transfer learning es viable en dominios médicos.

---

## 2. Pipeline de Datos Médicos

### 2.1 Preprocesamiento

Las imágenes médicas suelen venir en formatos DICOM. El pipeline debe:
1. Convertir a arrays numéricos (HU para CT, intensidad para RX).
2. Aplicar windowing (ajuste de contraste para tejidos específicos).
3. Redimensionar a una resolución estándar (ej. $224 \times 224$ o $512 \times 512$).
4. Normalizar por el valor medio y desviación estándar del dataset.

```python
import pydicom
from PIL import Image
import numpy as np

def preprocess_dicom(dcm_path, target_size=(512, 512)):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img).resize(target_size)
    return np.array(img)
```

⚠️ **Advertencia:** Nunca normalices con estadísticas de ImageNet directamente sobre radiografías. Las distribuciones de intensidad son radicalmente diferentes. Calcula la media y desviación de tu propio dataset médico.

---

## 3. Manejo del Desbalanceo

### 3.1 Pesos de Clase

Asignar mayor peso a la clase minoritaria en la función de pérdida:

$$
\mathcal{L} = -\sum_{i} w_{y_i} \, y_i \log(\hat{y}_i)
$$

En PyTorch:

```python
class_weights = torch.tensor([1.0, 5.0]).to(device)  # Clase positiva 5x más pesada
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 3.2 Oversampling

Utilizar `WeightedRandomSampler` para que el DataLoader muestree más frecuentemente la clase minoritaria:

```python
from torch.utils.data import WeightedRandomSampler

class_counts = [900, 100]  # Ejemplo de distribución
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### 3.3 Focal Loss

Modifica la cross-entropy para enfocarse en ejemplos difíciles:

$$
\mathcal{L}_{FL} = -\alpha_t (1 - \hat{y}_t)^{\gamma} \log(\hat{y}_t)
$$

Donde $\gamma \geq 0$ reduce la contribución de ejemplos fáciles. Es especialmente útil cuando la clase mayoritaria domina fácilmente.

Caso real: en la detección de metástasis en imágenes de patología digital (Whole Slide Images), el focal loss es estándar porque el tejido canceroso puede ocupar menos del 1% del área total.

💡 **Tip:** Si tienes un dataset con 95% negativos y 5% positivos, un modelo trivial que siempre predice negativo tendrá 95% de accuracy. Usa métricas como AUC-ROC, sensibilidad y especificidad en lugar de accuracy.

---

## 4. Augmentación Específica Médica

La augmentación en imágenes médicas debe ser conservadora para no alterar la morfología patológica:

| Transformación | Segura para médico | Razón |
|----------------|--------------------|-------|
| Rotación ±10° | Sí | La orientación del paciente varía |
| Volteo horizontal | Sí* | *No aplicar si hay asimetría clínica relevante (ej. cardiomegalia) |
| Volteo vertical | No | Anatomía estándar (cráneo arriba) |
| Cambio de brillo/contraste | Sí (moderado) | Simula variaciones de exposición |
| Zoom/Recorte | Sí | Simula diferentes campos de visión |
| Elastic deformation | Sí (subtle) | Simula variaciones de tejido blando |
| Gaussian noise | Sí (leve) | Robustecer contra ruido del sensor |

```python
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
```

⚠️ **Advertencia:** Una rotación de 180° en una radiografía de tórax puede ser indistinguible visualmente para un modelo, pero anatómicamente incorrecta. Siempre valida tus transformaciones con un experto clínico.

---

## 5. Métricas Clínicas

En medicina, los falsos negativos (no detectar una enfermedad) y los falsos positivos (alarma innecesaria) tienen costos muy diferentes.

### 5.1 Sensibilidad (Recall) y Especificidad

$$
\text{Sensibilidad} = \frac{TP}{TP + FN}
$$

$$
\text{Especificidad} = \frac{TN}{TN + FP}
$$

### 5.2 AUC-ROC

Mide la capacidad del modelo para discriminar entre clases a todos los umbrales posibles. Un AUC de 0.5 equivale a azar; 1.0 es perfecto.

### 5.3 F1-Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

| Métrica | ¿Qué mide? | Prioridad cuando... |
|---------|------------|---------------------|
| Sensibilidad | Capacidad de detectar enfermos | Importa minimizar falsos negativos (screening) |
| Especificidad | Capacidad de descartar sanos | Importa minimizar falsos positivos (evitar ansiedad/costos) |
| AUC-ROC | Discriminación global | Comparar modelos independientemente del umbral |
| F1 | Balance precisión-recall | Clases desbalanceadas |

---

## 6. Interpretabilidad con Grad-CAM

Los médicos no confiarán en una caja negra. Grad-CAM (Gradient-weighted Class Activation Mapping) genera un mapa de calor que resalta las regiones de la imagen que más influenciaron la predicción del modelo.

Para una clase $c$, el mapa de activación de clase es:

$$
L^c_{Grad-CAM} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

Donde $A^k$ son los feature maps de la última capa convolucional y $\alpha^c_k$ son los pesos obtenidos promediando los gradientes de la clase $c$ respecto a $A^k$:

$$
\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

En PyTorch:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.backbone.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=image_tensor, targets=None)
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
```

Caso real: en el sistema de detección de retinopatía diabética de Google Health, los mapas de Grad-CAM se utilizaron para validar que el modelo enfocaba su atención en microaneurismas y hemorragias retinianas, no en artefactos de imagen.

💡 **Tip:** Si Grad-CAM resalta bordes de la imagen o marcas de texto en lugar de anatomía patológica, tu modelo probablemente está aprendiendo artefactos del dataset (data leakage). Investiga inmediatamente.

---

## 7. Implementación del Pipeline Médico

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

class MedicalClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(MedicalClassifier, self).__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 📦 Código de Compresión

```python
"""
Script completo de clasificador médico con PyTorch.
Incluye: DenseNet-121 pre-entrenado, Focal Loss,
WeightedRandomSampler, métricas clínicas y Grad-CAM.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MedicalCNN, self).__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()

def get_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

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
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs)
    auc = roc_auc_score(all_labels, all_probs)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Negativo', 'Positivo']))
    print(f"AUC-ROC: {auc:.4f}")
    return all_labels, all_probs, all_preds

# Nota: Requiere dataset médico real o proxy (ej. ChestX-ray14).
# Aquí se muestra la estructura del pipeline completo.
```

---

## 🎯 Proyecto: Clasificador de Neumonía en Radiografías de Tórax

**Descripción:**
Desarrollar un sistema end-to-end para clasificar radiografías de tórax en dos categorías: NORMAL y NEUMONÍA. El proyecto simula un entorno clínico real donde la interpretabilidad y las métricas de sensibilidad son tan importantes como la accuracy. Se utilizará el dataset público ChestX-ray14 o un subset organizado de Kaggle como proxy.

**Requisitos funcionales:**
1. Implementar un pipeline de preprocesamiento que cargue imágenes en escala de grises, las redimensione a $224 \times 224$ y aplique normalización basada en estadísticas del dataset médico (no ImageNet, o al menos documentar la diferencia).
2. Construir un modelo basado en DenseNet-121 pre-entrenado, con fine-tuning de las últimas capas densas y del clasificador final.
3. Manejar el desbalanceo de clases mediante:
   a. `WeightedRandomSampler` en el DataLoader.
   b. Focal Loss como alternativa a CrossEntropy ponderada.
   c. Comparar resultados de ambas estrategias.
4. Aplicar augmentación médica conservadora: rotaciones pequeñas, volteo horizontal controlado, ajuste de brillo/contraste.
5. Entrenar con Adam y learning rate scheduling (ReduceLROnPlateau), implementando early stopping basado en AUC-ROC de validación.
6. Evaluar con métricas clínicas: sensibilidad, especificidad, precision, recall, F1-score y AUC-ROC.
7. Integrar Grad-CAM para generar mapas de calor de al menos 10 ejemplos de test, verificando que el modelo atiende a regiones pulmonares y no a artefactos.
8. Generar un informe ejecutivo en Markdown que documente: distribución de datos, arquitectura, hiperparámetros, curvas de entrenamiento, métricas finales y hallazgos de interpretabilidad.

**Componentes principales:**
- `data_pipeline.py`: carga, preprocesamiento, splits train/val/test y transforms médicos.
- `models.py`: `PneumoniaDenseNet` con soporte para congelar/descongelar capas.
- `losses.py`: implementaciones de `WeightedCrossEntropy` y `FocalLoss`.
- `train.py`: bucle de entrenamiento con logging de métricas, checkpointing y early stopping.
- `evaluate.py`: cálculo de métricas clínicas y generación de curvas ROC.
- `gradcam_visualizer.py`: wrapper sobre `pytorch-grad-cam` para producir overlays sobre radiografías.
- `report_generator.py`: ensambla gráficos y tablas en un informe Markdown.

**Métricas de éxito:**
- AUC-ROC en test $\geq 0.90$.
- Sensibilidad $\geq 0.85$ (minimizar falsos negativos en screening).
- Especificidad $\geq 0.80$ (evitar alarmas innecesarias).
- Grad-CAM debe mostrar activación concentrada en campos pulmonares en al menos 8 de 10 casos analizados.
- Diferencia AUC entre entrenamiento y test $< 0.05$ (buena generalización).

**Referencias:**
- Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- Wang, X., et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR.
