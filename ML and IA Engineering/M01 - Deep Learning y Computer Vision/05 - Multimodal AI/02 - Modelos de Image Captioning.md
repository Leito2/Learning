# 📝 Modelos de Image Captioning

La generación de descripciones automáticas de imágenes es una de las tareas más desafiantes de la inteligencia artificial multimodal. No basta con reconocer objetos; es necesario capturar relaciones espaciales, atributos visuales y contexto semántico, para luego expresarlos en lenguaje natural gramaticalmente correcto. Image captioning es, en esencia, un problema de traducción entre modalidades: de la "lengua" visual a la lengua textual.

## 1. De la Visión al Lenguaje: El Paradigma Encoder-Decoder

La arquitectura dominante desde 2015 es el **encoder-decoder**. El encoder (CNN o ViT) comprime la imagen en una representación latente de alto nivel. El decoder (RNN, LSTM o Transformer) genera la secuencia de palabras condicionada a dicha representación.

El porqué de esta separación es epistemológico: la CNN extrae *qué* está presente y *dónde*, mientras que el modelo de lenguaje modela *cómo* articularlo. Sin embargo, las primeras implementaciones (Vinyals et al., 2015) usaban un vector global único, perdiendo detalles espaciales críticos.

## 2. Atención Visual: Show, Attend and Tell

Xu et al. (2015) introdujeron el mecanismo de atención visual para captioning. En lugar de condicionar al decoder con un vector fijo, se condiciona con una representación espacial $a = \{a_1, \dots, a_L\}$, donde cada $a_i$ es un vector de características de una región de la imagen.

La atención en el paso $t$ del decoder se calcula como:

$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{L} \exp(e_{tj})}$$

donde los scores de alineación son:

$$e_{ti} = f_{att}(s_{t-1}, a_i)$$

tipicamente implementado como una red feedforward o producto escalar. El contexto visual es entonces:

$$c_t = \sum_{i=1}^{L} \alpha_{ti} a_i$$

Este vector $c_t$ se concatena con el estado oculto $s_{t-1}$ para predecir la siguiente palabra. La atención permite al modelo "mirar" diferentes regiones mientras genera cada palabra, introduciendo una forma temprana de explicabilidad.

## 3. Métricas de Evaluación

Evaluar captioning es intrínsecamente difícil porque múltiples descripciones válidas pueden existir para una imagen. Las métricas automáticas comparan el candidato contra referencias humanas.

### BLEU (Bilingual Evaluation Understudy)

Mide la precisión de n-gramas modificada. Para un candidato $c$ y un conjunto de referencias $R$:

$$\text{BLEU-N} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)$$

donde $p_n$ es la precisión de n-gramas:

$$p_n = \frac{\sum_{C \in \text{cand}} \sum_{n\text{-gram} \in C} \text{Count}_{clip}(n\text{-gram})}{\sum_{C \in \text{cand}} \sum_{n\text{-gram} \in C} \text{Count}(n\text{-gram})}$$

y BP (Brevity Penalty) penaliza descripciones demasiado cortas:

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{(1-r/c)} & \text{if } c \leq r \end{cases}$$

### Tabla Comparativa de Métricas

| Métrica | Qué mide | Fortaleza | Debilidad |
|---|---|---|---|
| **BLEU** | Precisión n-gram | Rápida, correlada con fluidez | Ignora sinonimia, fija en forma |
| **METEOR** | Alineación flexible (sinónimos, stemming) | Mejor correlación humana | Más lenta, parámetros de idioma |
| **ROUGE** | Recall n-gram (oriendo de summarization) | Captura cobertura | Menos usada en captioning puro |
| **CIDEr** | Consenso entre referencias (TF-IDF n-gram) | Diseñada para captioning | Sesgada por frecuencia de palabras |
| **SPICE** | Grafo semántico (objetos, relaciones) | Captura significado | Requiere parsing, computacionalmente cara |

## 4. Estrategias de Generación: Beam Search vs Nucleus Sampling

Durante la inferencia, el decoder genera palabras de forma autoregresiva. Las estrategias de decodificación afectan drásticamente la calidad.

- **Beam Search**: Mantiene las $k$ secuencias más probables en cada paso. Maximiza la probabilidad conjunta, pero tiende a producir lenguaje genérico ("a person is doing something").
- **Nucleus Sampling (top-p)**: Muestrea de entre los tokens más probables cuya masa de probabilidad acumulada supera $p$. Introduce diversidad y creatividad, a costa de menor coherencia ocasional.

El porqué de la diferencia radica en la función objetivo: beam search optimiza la esperanza de la log-verosimilitud, mientras que nucleus sampling optimiza la percepción humana de interesante/novedoso.

## 5. Modelos Modernos: BLIP, BLIP-2 y LLaVA

Los paradigmas modernos han abandonado los LSTM en favor de Transformers y modelos fundacionales.

- **BLIP**: Propone un framework de bootstrap con generación de captions sintéticos y filtrado de ruido. Usa un Multimodal mixture of Encoder-Decoder (MED).
- **BLIP-2**: Introduce un Q-Former como puente entre un image encoder congelado (ViT) y un LLM congelado. El Q-Former aprende "consultas" latentes que resumen la información visual más relevante para el LLM.
- **LLaVA**: Conecta directamente el proyector visual de CLIP con LLaMA mediante una simple capa MLP, demostrando que con datos de instrucciones de alta calidad, no se necesitan arquitecturas complejas de fusión.

Caso real: **Microsoft Azure AI Vision** ofrece API de image captioning entrenada con millones de pares imagen-texto, soportando descripciones en múltiples idiomas.

Caso real: **Google Cloud Vision API** incluye generación de descripciones detalladas para accesibilidad web (atributos alt automáticos).

⚠️ **Advertencias**

- **Evaluación sobreajustada a BLEU**: Optimizar directamente BLEU raramente produce captions humanamente mejores. Es una métrica de proxy imperfecta.
- **Object hallucination**: Los modelos generan objetos que no existen en la imagen (e.g., "un sombrero" en una persona descalza). Es un problema activo de investigación.
- **Sesgo de género y raza**: Los datasets de captioning (COCO, etc.) contienen sesgos demográficos que los modelos amplifican.

💡 **Tips y Reglas Mnemotécnicas**

- **"BLEU mide clones"**: Si tu caption copia palabras exactas de la referencia, BLEU sube. No confíes ciegamente en ella.
- **"Atención = Foco de cámara"**: Cuando el modelo escribe "un gato", la atención debería pesar las regiones felinas. Visualízala para debuggear.
- **"BLIP limpia datos"**: Si tu dataset es ruidoso, BLIP puede generar captions sintéticos de mejor calidad que los humanos anotadores rándom.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualAttention(nn.Module):
    def __init__(self, hidden_dim, feature_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(feature_dim, hidden_dim)
        self.W_att = nn.Linear(hidden_dim, 1)
    
    def forward(self, features, hidden):
        # features: (B, L, D), hidden: (B, H)
        hidden = hidden.unsqueeze(1)  # (B, 1, H)
        att = torch.tanh(self.W_v(features) + self.W_h(hidden))
        scores = self.W_att(att).squeeze(-1)  # (B, L)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return context, weights
```

📦 **Código de Compresión PyTorch**

```python
"""
Script compresivo: Encoder-Decoder con Atención Visual para Captioning.
Resume: CNN encoder, LSTM decoder, atención, beam search dummy.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, encoded_size=14, feature_dim=512):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        self.proj = nn.Linear(2048, feature_dim)
    
    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, H, W)
        features = self.adaptive(features)
        features = features.permute(0, 2, 3, 1)  # (B, H, W, 2048)
        features = features.view(features.size(0), -1, features.size(-1))
        return self.proj(features)  # (B, L, feature_dim)

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()
        self.fc_feature = nn.Linear(feature_dim, attention_dim)
        self.fc_hidden = nn.Linear(hidden_dim, attention_dim)
        self.fc_att = nn.Linear(attention_dim, 1)
    
    def forward(self, features, hidden):
        att = self.fc_att(torch.tanh(self.fc_feature(features) + 
                                     self.fc_hidden(hidden).unsqueeze(1)))
        alpha = torch.softmax(att.squeeze(2), dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, feature_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(feature_dim, hidden_dim, 256)
        self.lstm = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        h = self.init_h(features.mean(dim=1))
        c = self.init_c(features.mean(dim=1))
        embeddings = self.embedding(captions)
        outputs = []
        for t in range(captions.size(1)):
            context, _ = self.attention(features, h)
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            out = self.fc(h)
            outputs.append(out)
        return torch.stack(outputs, dim=1)

# Uso dummy
if __name__ == "__main__":
    encoder = EncoderCNN(feature_dim=512)
    decoder = DecoderRNN(embed_dim=256, feature_dim=512, hidden_dim=512, vocab_size=10000)
    imgs = torch.randn(2, 3, 224, 224)
    caps = torch.randint(0, 10000, (2, 20))
    feats = encoder(imgs)
    outs = decoder(feats, caps)
    print("Output shape:", outs.shape)
```

🎯 **Proyecto: Generador de Descripciones Accesibles para E-commerce**

**Descripción**: Sistema que genera captions detallados y SEO-friendly para catálogos de productos de moda, incluyendo atributos visuales (color, textura, estilo).

**Requisitos funcionales**:
1. Ingesta de imágenes de producto en alta resolución.
2. Extracción de features visuales mediante ViT.
3. Generación de captions con modelo Transformer decoder.
4. Filtrado post-hoc para evitar alucinaciones (lista de atributos permitidos).
5. Exportación de metadata en formato JSON para CMS.

**Componentes principales**:
- Modelo de captioning fine-tuned en dataset de productos.
- Módulo de extracción de atributos visuales (colores dominantes).
- Pipeline de validación gramatical.

**Métricas de éxito**:
- CIDEr > 1.0 en validación interna.
- Tasa de alucinación de objetos < 5 %.
- Aprobación humana > 85 %.

**Referencias**:
- Xu et al., "Show, Attend and Tell", ICML 2015.
- Li et al., "BLIP: Bootstrapping Language-Image Pre-training", ICML 2022.
