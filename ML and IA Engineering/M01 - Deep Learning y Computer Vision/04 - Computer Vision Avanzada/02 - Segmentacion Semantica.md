# 🔲 Segmentación Semántica

La segmentación semántica eleva la comprensión de imágenes al nivel del píxel. En lugar de responder "¿qué hay?" o "¿dónde está?", responde "¿qué clase corresponde a cada píxel?". Esta granularidad es esencial para aplicaciones donde los contornos exactos importan: cirugía asistida, vehículos autónomos, análisis de imágenes satelitales y edición de imágenes.

---

## 1. Clasificación por píxel: el problema fundamental

Formalmente, dada una imagen $I \in \mathbb{R}^{H \times W \times 3}$, buscamos una función $f$ que asigne una etiqueta de clase $c \in \{1, \dots, C\}$ a cada píxel:

$$
S = f(I) \in \{1, \dots, C\}^{H \times W}
$$

**Diferencias con detección:**
- La detección produce boxes rectangulares que incluyen fondo.
- La segmentación delimita el contorno exacto del objeto.
- La **instance segmentation** va más allá: diferencia entre "persona 1" y "persona 2".

Caso real: **Tesla Autopilot** utiliza segmentación semántica para delimitar el espacio drivable, marcas viales y objetos, combinándolo con detección para planificación de trayectoria.

---

## 2. Fully Convolutional Networks (FCN)

Long et al. (2015) propusieron reemplazar las capas fully connected de una CNN por capas convolucionales $1 \times 1$, permitiendo que la entrada sea de cualquier tamaño y produciendo un mapa de calor espacial.

### 2.1 Encoder-decoder básico

El encoder es una CNN estándar (VGG, ResNet) que reduce progresivamente la resolución espacial y aumenta la profundidad semántica. El problema: la salida es mucho más pequeña que la entrada.

Para recuperar la resolución, FCN utiliza **transposed convolutions** (a veces llamadas deconvolutions, aunque matemáticamente son convoluciones transpuestas):

$$
y[i, j] = \sum_{m,n} x[i + m, j + n] \cdot h[m, n]
$$

### 2.2 Skip connections en FCN

FCN-16 y FCN-8 combinan capas de distintas profundidades del encoder con el decoder. Esto preserva detalles finos de bordes que se pierden en las capas profundas.

💡 **Tip**: los skip connections funcionan porque las capas tempranas capturan información de bajo nivel (bordes, texturas) y las capas profundas capturan semántica de alto nivel (objetos completos). Combinar ambas mejora los contornos.

---

## 3. U-Net: la revolución biomédica

Ronneberger et al. (2015) diseñaron U-Net para segmentación de imágenes médicas con pocos datos de entrenamiento.

### 3.1 Arquitectura simétrica

U-Net consiste en:
- **Encoder (contracción)**: bloques convolucionales + max pooling que reducen resolución.
- **Bottleneck**: capa más profunda con mayor contexto.
- **Decoder (expansión)**: upsampling + convoluciones que restauran resolución.
- **Skip connections**: concatenación de features del encoder con el decoder en cada nivel.

La forma de "U" en el diagrama de arquitectura le da el nombre.

### 3.2 ¿Por qué funciona tan bien con pocos datos?

La información espacial de alta frecuidad (bordes) se propaga directamente desde el encoder al decoder mediante concatenación. Esto permite que el decoder reconstruya contornos precisos sin depender exclusivamente de la información comprimida del bottleneck.

Caso real: **Radiología computarizada** usa U-Net para segmentar tumores cerebrales en resonancias magnéticas. Con tan solo 30 imágenes etiquetadas, U-Net supera métodos clásicos debido a su fuerte prior de forma.

⚠️ **Advertencia**: el tamaño de entrada de U-Net debe ser múltiplo de $2^N$, donde $N$ es el número de capas de pooling. Si no, los skip connections no alinean espacialmente.

---

## 4. DeepLab: Atrous Convolution y ASPP

### 4.1 Atrous (Dilated) Convolution

El problema del encoder-decoder es que el max pooling destruye información espacial. DeepLab lo evita utilizando **atrous convolutions** que aumentan el campo receptivo sin perder resolución.

Definición 1D:

$$
y[i] = \sum_{k=1}^{K} x[i + r \cdot k] \cdot h[k]
$$

donde $r$ es la **tasa de dilatación**. Cuando $r=1$, es convolución estándar. A medida que $r$ crece, el campo receptivo crece exponencialmente.

| Tasa | Campo receptivo efectivo | Uso típico |
|------|--------------------------|------------|
| 1 | $3 \times 3$ | Detalle fino, bordes |
| 6 | $13 \times 13$ | Contexto medio |
| 12 | $25 \times 25$ | Contexto global |

### 4.2 Atrous Spatial Pyramid Pooling (ASPP)

DeepLabv3+ aplica múltiples atrous convolutions en paralelo con distintas tasas y concatena los resultados. Esto captura objetos a múltiples escalas simultáneamente.

$$
\text{ASPP}(x) = \text{Concat}\left[ \text{Conv}_{r=1}(x), \text{Conv}_{r=6}(x), \text{Conv}_{r=12}(x), \text{Conv}_{r=18}(x), \text{GAP}(x) \right]
$$

donde GAP es Global Average Pooling para capturar contexto global.

Caso real: **Google Street View** utiliza DeepLab para segmentar carreteras, aceras y edificios en imágenes panorámicas, donde la escala de los objetos varía drásticamente con la perspectiva.

---

## 5. Mask R-CNN: Instance Segmentation

He et al. (2017) extendieron Faster R-CNN para producir máscaras de segmentación por instancia.

### 5.1 Arquitectura

1. La imagen pasa por un backbone + FPN.
2. La RPN propone ROIs.
3. Cada ROI se alinea espacialmente con **ROIAlign** (bilineal, sin cuantización).
4. Paralelamente se predicen: clase, box refinado y máscara binaria $28 \times 28$.

ROIAlign es crítico: ROI Pooling cuantiza coordenadas, introduciendo desalineaciones de pocos píxeles que arruinan la máscara.

### 5.2 Pérdida multitarea

$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}
$$

La pérdida de máscara es entropía cruzada binaria por píxel y por clase, evitando competencia entre clases en la máscara.

💡 **Tip**: Mask R-CNN es la elección por defecto cuando necesitas **tanto** bounding boxes **como** máscaras. Si solo necesitas máscaras semánticas (sin distinguir instancias), U-Net o DeepLab son más eficientes.

---

## 6. Métricas de segmentación

### 6.1 Pixel Accuracy

$$
\text{Pixel Accuracy} = \frac{\sum_{i} \mathbb{1}(y_i = \hat{y}_i)}{N}
$$

Es engañosa cuando hay desbalance de clases (por ejemplo, fondo ocupa el 90 % de la imagen).

### 6.2 Intersection over Union (IoU) / Jaccard Index

Por clase:

$$
\text{IoU}_c = \frac{|Y_c \cap \hat{Y}_c|}{|Y_c \cup \hat{Y}_c|}
$$

**mIoU**: promedio sobre clases.

### 6.3 Dice Coefficient (F1-score espacial)

$$
\text{Dice}_c = \frac{2 |Y_c \cap \hat{Y}_c|}{|Y_c| + |\hat{Y}_c|}
$$

Relación con IoU:

$$
\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}
$$

El Dice penaliza más los falsos positivos y negativos en clases pequeñas, siendo preferido en imágenes médicas.

⚠️ **Advertencia**: no uses Pixel Accuracy en datasets desbalanceados. Usa mIoU o Dice. En segmentación médica, el Dice es el estándar de facto.

---

## 📦 Código de compresión

```python
"""
Segmentación Semántica completa con PyTorch.
Incluye U-Net básico, métricas IoU/Dice y entrenamiento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._block(feature * 2, feature))

        self.bottleneck = self._block(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx + 1](x)
        return self.final_conv(x)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

def dice_coefficient(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print("Input:", x.shape, "Output:", preds.shape)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    print("Dice:", dice_coefficient(preds, target).item())
    print("IoU:", iou(preds, target).item())
```

---

## 🎯 Proyecto documentado: Segmentación de Tejido Tumoral en Histopatología

### Descripción
Sistema que analiza láminas de histopatología digitalizada (WSI) para segmentar regiones de tejido tumoral, estroma y región normal, asistiendo a patólogos en diagnóstico de cáncer.

### Requisitos funcionales
1. Procesar parches de $512 \times 512$ píxeles extraídos de WSI a 20x de magnificación.
2. Segmentar múltiples clases de tejido: tumor, estroma, núcleos, fondo.
3. Manejar variaciones de tinción (H&E) mediante augmentación o normalización.
4. Reconstruir la máscara completa del WSI a partir de las predicciones por parche.
5. Exportar máscaras en formato compatible con visores de patología digital (DICOM/GeoJSON).

### Componentes principales
- **Backbone**: U-Net con ResNet-34 preentrenado en ImageNet como encoder.
- **Preprocesamiento**: Normalización Macenko o Reinhard para estandarizar tinción H&E.
- **Postprocesamiento**: Reconstrucción de mosaico con suavizado de bordes (blending).
- **Infraestructura**: Pipeline en Dask para procesamiento distribuido de WSI grandes (>100k x 100k píxeles).

### Métricas de éxito
- mIoU > 0.80 en conjunto de validación de múltiples hospitales.
- Dice por clase tumor > 0.85.
- Concordancia inter-observador (patólogo vs modelo) > 0.90 en índice Kappa.
- Procesamiento de un WSI completo < 5 minutos en GPU.

### Referencias
- Long, J., Shelhamer, E., & Darrell, T. "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015.
- Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- Chen, L.-C., et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV 2018.
- He, K., et al. "Mask R-CNN." ICCV 2017.
