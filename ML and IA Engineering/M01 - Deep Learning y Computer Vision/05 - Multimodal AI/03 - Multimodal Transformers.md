# 🤖 Multimodal Transformers

Los Transformers revolucionaron el procesamiento del lenguaje natural mediante el mecanismo de self-attention. Su extensión al dominio multimodal no es una simple adaptación: es una reingeniería de cómo las señales heterogéneas interactúan. La pregunta central es si debemos fusionar las modalidades tempranamente (a nivel de token) o mantener flujos separados con intercambios controlados. Los modelos multimodales modernos exploran todo este espectro.

## 1. Self-Attention vs Cross-Attention

En self-attention, las queries (Q), keys (K) y values (V) provienen de la misma secuencia:

$$\text{Self-Att}(X) = \text{softmax}\left(\frac{Q_X K_X^T}{\sqrt{d_k}}\right) V_X$$

En **cross-attention**, las queries provienen de una modalidad destino (e.g., texto) mientras que las keys y values provienen de una modalidad fuente (e.g., imagen):

$$\text{Cross-Att}(Q_{text}, K_{img}, V_{img}) = \text{softmax}\left(\frac{Q_{text} K_{img}^T}{\sqrt{d_k}}\right) V_{img}$$

El porqué de la separación es clave: self-attention modela relaciones intra-modales (e.g., qué palabra depende de qué otra palabra, o qué parche de imagen está relacionado con otro). Cross-attention modela relaciones **inter-modales**, permitiendo que el texto "consulte" directamente la información visual. Arquitecturas como DETR y los decoders de captioning usan cross-attention para condicionar la generación textual en contenido visual.

## 2. Modelos Encoder-Only: ALBEF

ALBEF (Align before Fuse) desafía la fusión temprana. Propone un modelo de visión (ViT) y un modelo de lenguaje (BERT) separados, con un encoder multimodal sobre ellos. La innovación es un objetivo de alineación contrastiva *antes* de la fusión.

La hipótesis es que alinear las representaciones unimodales en un espacio común *previo* a la interacción cross-modal estabiliza el entrenamiento y mejora la interpretabilidad. ALBEF utilita momentum distillation para generar pseudo-objetivos de una versión en progreso del modelo, mitigando el ruido en los datos web.

## 3. Modelos Encoder-Decoder: BLIP

BLIP (Bootstraping Language-Image Pre-training) unifica comprensión y generación en un solo framework. Su arquitectura MED (Multimodal Encoder-Decoder) soporta tres funciones:

1. **Image-text contrastive learning** (encoder unimodal).
2. **Image-text matching** (encoder multimodal con cross-attention).
3. **Image captioning** (decoder con cross-attention).

El decoder de BLIP genera texto auto-regresivamente, condicionado en la imagen mediante cross-attention. Esto permite que el mismo modelo pre-entrenado sirva tanto para retrieval (encoder) como para captioning (decoder).

## 4. Flamingo y el In-Context Learning Visual

Flamingo (DeepMind) es un modelo de few-shot learning visual que procesa secuencias intercaladas de imágenes y texto. Su arquitectura inserta bloques de **GATED CROSS-ATTENTION** densamente en capas de un LLM pre-entrenado, sin modificar sus pesos de self-attention.

La idea profunda es que el conocimiento lingüístico ya capturado por el LLM puede ser "condicionado" por información visual a través de capas de adaptación mínimas. Flamingo demuestra que la multimodalidad puede lograrse via **in-context learning**, donde ejemplos de pares (imagen, respuesta) en el prompt guían al modelo, analogamente a GPT-3 en texto puro.

## 5. GPT-4V y Gemini

- **GPT-4V (Visión)**: Extiende GPT-4 para aceptar imágenes como entrada. Aunque los detalles arquitectónicos son propietarios, se presume que utiliza un sistema de tokenización visual (posiblemente con un encoder de visión y un proyector) alimentando un decoder de lenguaje masivo. Su fortaleza es el razonamiento: puede resolver problemas matemáticos escritos en papel, interpretar diagramas técnicos y seguir instrucciones complejas en imagen+texto.
- **Gemini (Google)**: Diseñado desde cero como modelo nativamente multimodal. Se entrena conjuntamente sobre texto, imagen, audio y video. La arquitectura utiliza sparse Mixture-of-Experts (MoE) y atención multimodal nativa, buscando que las modalidades no sean "add-ons" sino ciudadanos de primera clase del espacio de representación.

## 6. Arquitecturas Unificadas

La tendencia actual apunta a **modelos unificados** donde una sola red procesa cualquier modalidad. Ejemplos incluyen:

- **Data2Vec**: Framework unificado para habla, visión y NLP.
- **ImageBind (Meta)**: Alinea seis modalidades (imagen, texto, audio, profundidad, térmico, IMU) en un único espacio usando pares con imagen como puente.
- **Unified-IO**: Tarea cualquier problema de visión y lenguaje con una sola arquitectura encoder-decoder.

El porqué de la unificación es la ley de escalamiento: múltiples modalidades proveen señales de supervisión mutua que mejoran la generalización y robustez.

## 7. Tokenización de Imágenes

Para que un Transformer procese imágenes, deben convertirse en secuencias. Los métodos dominantes son:

- **Patch Embedding (ViT)**: La imagen se divide en parches no solapados $P \times P$, se aplanan y se proyectan linealmente. Para una imagen $224 \times 224$ y parches $16 \times 16$, se obtienen $L = 196$ tokens visuales.
- **Pixel-level**: Modelos como iGPT trabajan directamente sobre píxeles, pero son computacionalmente prohibitivos.
- **Perceptual VQ-VAE**: Modelos como DALL-E y BEiT usan un VQ-VAE para tokenizar imágenes en un vocabulario discreto de "tokens visuales", permitiendo que el Transformer opere como si fuera un modelo de lenguaje sobre un vocabulario mixto.

Caso real: **Google Gemini 1.5 Pro** procesa secuencias de hasta 1 millón de tokens, permitiendo analizar videos largos junto con texto, gracias a una tokenización eficiente y arquitectura de long contexto.

Caso real: **Meta ImageBind** permite consultas como "encontrar un video que suene como esto" alineando audio e imagen sin pares de entrenamiento directos.

⚠️ **Advertencias**

- **Cuellos de botella en cross-attention**: La atención cruzada tiene complejidad $O(T \cdot L)$ donde $T$ es longitud de texto y $L$ número de parches visuales. Para imágenes de alta resolución, esto explota rápidamente.
- **Desalineación modal en modelos congelados**: Congelar el LLM y añadir adapters (como en BLIP-2) puede limitar la capacidad de razonamiento multimodal profundo.
- **Positional bias**: Los transformers visuales pierden la inductive bias de convolución (localidad, traslación). Requieren datasets masivos para compensar.

💡 **Tips y Reglas Mnemotécnicas**

- **"Self = mí mismo, Cross = cruzado"**: Self-attention mira dentro de su propia modalidad; cross-attention cruza la mirada hacia otra.
- **"Gemini es gemelo"**: Nace junto, multimodal desde el vientre. No es texto con gafas de visión.
- **"Flamingo rosa, adapta poco"**: Añade solo cross-attention layers; no toca el LLM base. Menos parámetros entrenables, más few-shot.

```python
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
    
    def forward(self, queries, kv):
        # queries: (B, T, D_q), kv: (B, L, D_kv)
        B, T, _ = queries.shape
        _, L, _ = kv.shape
        
        Q = self.q_proj(queries)
        K = self.k_proj(kv)
        V = self.v_proj(kv)
        
        # Multi-head split
        Q = Q.view(B, T, self.num_heads, -1).transpose(1, 2)
        K = K.view(B, L, self.num_heads, -1).transpose(1, 2)
        V = V.view(B, L, self.num_heads, -1).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
```

📦 **Código de Compresión PyTorch**

```python
"""
Script compresivo: Transformer Multimodal con Cross-Attention.
Resume: patch embedding, encoder ViT, decoder Transformer con cross-attn.
"""
import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
            for _ in range(depth)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultimodalDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, dim, depth, heads):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, embed_dim))
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
            for _ in range(depth)
        ])
        self.self_attn = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=heads, batch_first=True)
            for _ in range(depth)
        ])
        self.fc_out = nn.Linear(dim, vocab_size)
    
    def forward(self, tokens, memory):
        x = self.token_emb(tokens) + self.pos_emb[:, :tokens.size(1), :]
        for ca, sa in zip(self.cross_attn, self.self_attn):
            # Cross-attention a memoria visual
            ca_out, _ = ca(x, memory, memory)
            x = x + ca_out
            # Self-attention causal
            x = sa(x, x)
        return self.fc_out(x)

class SimpleMultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, img_size=224, patch_size=16, dim=768):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=dim)
        self.img_encoder = TransformerEncoder(dim, depth=6, heads=12)
        self.decoder = MultimodalDecoder(vocab_size, dim, dim, depth=6, heads=12)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, images, caption_tokens):
        patches = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(patches.size(0), -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        memory = self.img_encoder(patches)
        return self.decoder(caption_tokens, memory)

if __name__ == "__main__":
    model = SimpleMultimodalTransformer(vocab_size=10000)
    imgs = torch.randn(2, 3, 224, 224)
    toks = torch.randint(0, 10000, (2, 20))
    out = model(imgs, toks)
    print("Output shape:", out.shape)
```

🎯 **Proyecto: Asistente de Documentación Técnica Multimodal**

**Descripción**: Sistema que ingiere manuales técnicos (PDF con diagramas) y permite consultas en lenguaje natural sobre el contenido visual y textual.

**Requisitos funcionales**:
1. Extracción de imágenes y texto de documentos PDF.
2. Tokenización de imágenes en parches y texto en subword tokens.
3. Indexación conjunta en un modelo de cross-attention.
4. Respuesta a preguntas tipo: "¿Qué componente está señalado en la Figura 3?".
5. Generación de resúmenes multimodales.

**Componentes principales**:
- Parser de PDF (pymupdf).
- Modelo multimodal tipo BLIP-2 o LLaVA.
- Base de datos vectorial para retrieval de páginas.

**Métricas de éxito**:
- Exactitud de grounding visual > 80 %.
- BLEU-4 de respuestas > 25.
- Latencia de respuesta < 3 s por página.

**Referencias**:
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022.
- Team et al., "Gemini: a Family of Highly Capable Multimodal Models", arXiv 2023.
