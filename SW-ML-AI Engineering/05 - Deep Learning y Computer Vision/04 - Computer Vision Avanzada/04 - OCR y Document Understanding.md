# 📄 OCR y Document Understanding

El reconocimiento óptico de caracteres (OCR) es uno de los problemas más antiguos de la visión por computadora, pero su combinación con deep learning y comprensión documental ha creado un campo completamente nuevo: **Document AI**. En lugar de simplemente "leer" texto, los sistemas modernos entienden la estructura espacial, las relaciones semánticas y el contexto de negocio en documentos escaneados, facturas, contratos y formularios.

---

## 1. OCR tradicional vs Deep Learning

### 1.1 OCR clásico

Los sistemas tradicionales utilizan pipelines en cascada:
1. **Preprocesamiento**: binarización, deskewing, eliminación de ruido.
2. **Detección de texto**: segmentación de componentes conectados, proyección de perfiles.
3. **Reconocimiento**: extracción de features manuales (HOG, momentos) + clasificador (SVM, k-NN).

**Problemas**: fragilidad ante variaciones de fuente, rotación, fondo complejo, idiomas mixtos.

### 1.2 Deep Learning para OCR

El paradigma moderno unifica detección y reconocimiento mediante redes neuronales:
- **Detección**: redes convolucionales o transformers predicen cajas de texto.
- **Reconocimiento**: secuencias de caracteres se modelan con CTC (Connectionist Temporal Classification) o atención.

Caso real: **Adobe Acrobat Pro** utiliza modelos de deep learning para OCR en PDF escaneados, superando significativamente a motores tradicionales en documentos con layouts complejos.

![Documento de prueba para OCR](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Test_OCR_document.jpg/640px-Test_OCR_document.jpg)

---

## 2. Detección de texto: EAST

Zhou et al. (2017) propusieron **EAST (Efficient and Accurate Scene Text detection)**, una red fully convolutional que predice palabras o líneas de texto en tiempo real.

### 2.1 Arquitectura

- **Backbone**: PVANet o ResNet-50.
- **Decoder**: fusiona features de múltiples escalas (U-Net style).
- **Salida**: para cada píxel predice:
  - **Score map**: probabilidad de pertenecer a texto.
  - **Geometry map**: distancias a los 4 bordes del box rotado (RBOX) o 8 coordenadas (QUAD).

### 2.2 Ventajas clave

EAST elimina los componentes intermedios (propuesta de región, filtrado, refinemiento) y produce boxes directamente. Es **end-to-end trainable** y funciona a tiempo real.

```mermaid
flowchart TD
    A[Imagen] --> B[EAST<br/>Detección de texto]
    B --> C[CRNN<br/>Reconocimiento]
    C --> D[CTC<br/>Alineación]
    D --> E[Texto final]
```

⚠️ **Advertencia**: EAST puede fusionar líneas de texto cercanas si el umbral de NMS no está bien calibrado. En documentos densos, modelos basados en transformers (como LayoutLMv3) suelen ser más robustos.

---

## 3. Reconocimiento de texto: CRNN + CTC

Shi et al. (2016) propusieron **CRNN (Convolutional Recurrent Neural Network)**, que combina CNN, RNN y CTC en una arquitectura unificada.

### 3.1 Pipeline CRNN

1. **CNN**: extrae features visuales de la imagen de texto. Entrada: imagen de altura fija, ancho variable.
2. **RNN (Bidirectional LSTM)**: modela dependencias secuenciales en la dimensión horizontal.
3. **CTC**: alinea la secuencia de predicciones con la etiqueta sin requerer segmentación explícita por carácter.

### 3.2 CTC: matemática

Dada una secuencia de predicciones $\pi$ de longitud $T$ y una etiqueta $l$, CTC define una función de colapso $B$ que elimina repeticiones y el token blank $\epsilon$:

$$
B(aab\epsilon\epsilon c) = abc
$$

La probabilidad condicional es:

$$
P(l \mid x) = \sum_{\pi \in B^{-1}(l)} P(\pi \mid x) = \sum_{\pi \in B^{-1}(l)} \prod_{t=1}^{T} y_{\pi_t}^t
$$

El algoritmo **forward-backward** calcula esta suma eficientemente en $O(T \cdot |l|)$.

💡 **Tip**: CTC es ideal cuando la longitud de la secuencia de salida es menor o igual a la de entrada (como en texto horizontal). Para textos con layouts 2D complejos, la atención funciona mejor.

---

## 4. Transformers para OCR: TrOCR

Li et al. (2021) propusieron **TrOCR (Transformer-based Optical Character Recognition)**, que aplica un encoder-decoder transformer puro al OCR.

### 4.1 Arquitectura

- **Encoder**: ViT (DeiT) que procesa la imagen del texto como parches.
- **Decoder**: GPT-2 que autoregresivamente genera la secuencia de caracteres.

**Ventaja**: no requiere CTC ni componentes recurrentes. La atención cruzada entre el encoder visual y el decoder textual captura relaciones complejas entre la imagen y el lenguaje.

Caso real: **Microsoft Azure Form Recognizer** utiliza variantes de TrOCR para reconocimiento de texto manuscrito y mecanografiado en formularios, superando a CRNN en texto degradado.

---

## 5. Layout Analysis: más allá del texto

El layout de un documento (dónde están los títulos, tablas, párrafos, firmas) es tan importante como el texto mismo.

### 5.1 Detección de regiones de layout

Se puede modelar como detección de objetos donde las clases son:
- Título, párrafo, lista, tabla, figura, pie de página, etc.

Modelos como **Detectron2** con Mask R-CNN entrenados en datasets como PubLayNet logran mAP > 0.90 en detección de regiones.

### 5.2 Tablas en documentos

Las tablas son estructuras 2D donde las relaciones espaciales importan. Los enfoques modernos combinan:
- **Detección de celdas**: cada celda es un objeto.
- **Reconocimiento de estructura**: predicción de relaciones "arriba-de", "izquierda-de" entre celdas.
- **HTML/Excel generation**: decodificación de la estructura en formato estructurado.

⚠️ **Advertencia**: las tablas sin bordes visibles (tablas "invisibles") son el caso más difícil. Los modelos puros de visión fallan; se requiere entender el espaciado y alineación del texto.

---

## 6. Document AI: LayoutLM y familia

### 6.1 LayoutLMv1 (Xu et al., 2020)

La idea central: un documento es multimodal. Además del texto, la posición espacial de cada palabra en la página contiene información semántica.

**Entradas del modelo**:
- **Text embeddings**: de WordPiece (BERT).
- **Layout embeddings**: coordenadas normalizadas $(x_0, y_0, x_1, y_1)$ de cada bounding box.
- **Image embeddings**: opcional, features visuales de Faster R-CNN.

$$
E = E_{\text{text}} + E_{\text{layout}} + E_{\text{image}}
$$

Pretraining tasks:
- Masked visual-language modeling (MVLM).
- Multi-label document classification.

### 6.2 LayoutLMv2/v3 y Donut

- **LayoutLMv2**: incorpora atención espacial explícita y image embeddings en cada capa.
- **LayoutLMv3**: utiliza un ViT puro para el stream visual, eliminando la necesidad de Faster R-CNN.
- **Donut**: modelo OCR-free que recibe la imagen directamente y genera JSON estructurado mediante un decoder transformer.

Caso real: **Amazon Textract** y **Google Document AI** utilizan arquitecturas derivadas de LayoutLM para extraer pares key-value de facturas y recibos con precisión comercial.

```mermaid
flowchart TD
    A[Imagen] --> B[Text Embeddings]
    A --> C[Layout Embeddings]
    A --> D[Image Embeddings]
    B --> E[Suma<br/>E = E_text + E_layout + E_image]
    C --> E
    D --> E
    E --> F[Transformer Encoder]
    F --> G[NER / Key-Value]
```

💡 **Regla mnemotécnica**: **"Texto dice qué, layout dice dónde, imagen dice cómo"**. Los modelos de Document AI fusionan estas tres señales.

---

## 7. Key-Value Extraction

En documentos como facturas o contratos, la información está organizada en pares **campo-valor** (ej: "Total: $150.00").

### 7.1 Enfoques

1. **Sequence labeling**: BIO tagging sobre tokens de texto (B-BEGIN, I-INSIDE, O-OUTSIDE).
2. **Span extraction**: predicción de inicio y fin de cada valor.
3. **Generative**: el modelo genera directamente el JSON de salida (Donut).

### 7.2 Métricas

- **Field-level F1**: se considera correcto un campo si el texto extraído coincide exactamente (o con tolerancia) con el ground truth.
- **Tree-Edit Distance**: para estructuras jerárquicas como tablas anidadas.

⚠️ **Advertencia**: en key-value extraction, un pequeño error de OCR (por ejemplo, leer "S" como "$") arruina toda la extracción. Usa validación de formato (regex) post-OCR.

---

## 📦 Código de compresión

```python
"""
OCR y Document Understanding con PyTorch + Hugging Face.
Resume OCR con TrOCR y layout analysis con LayoutLMv3.
"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# --- Parte 1: OCR con TrOCR ---

def ocr_with_trocr(image_path):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# --- Parte 2: Layout + NER con LayoutLMv3 ---

def layout_ner(image_path, words, boxes):
    """
    words: lista de palabras detectadas por OCR.
    boxes: lista de bounding boxes [x0, y0, x1, y1] en escala 0-1000.
    """
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    encoding = processor(image_path, words, boxes=boxes, return_tensors="pt",
                         truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze())
    return list(zip(tokens, predictions))

if __name__ == "__main__":
    # Ejemplo OCR:
    # text = ocr_with_trocr("invoice.png")
    # print(text)

    # Ejemplo Layout NER (requiere OCR previo para words y boxes):
    # result = layout_ner("invoice.png", ["Total", "$100"], [[100,100,200,120],[210,100,280,120]])
    # print(result)
    print("Pipeline OCR + Document Understanding listo.")
```


