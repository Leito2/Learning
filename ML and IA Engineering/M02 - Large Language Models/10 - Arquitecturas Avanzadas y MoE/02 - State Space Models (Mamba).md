# 🐍 State Space Models: De RNN a Mamba

La auto-atención de los Transformers, aunque poderosa, impone una complejidad cuadrática $O(L^2)$ respecto a la longitud de la secuencia. Esta barrera limita severamente la aplicación de LLMs en dominios que requieren contextos extremadamente largos: genómica (secuencias de ADN de millones de pares de bases), audio (señales de alta frecuencia), análisis de video y documentos legales. Los **State Space Models (SSM)** representan un paradigma alternativo que recupera la complejidad lineal $O(L)$ de las RNN clásicas, pero con la capacidad de entrenamiento paralelo de los Transformers. En esta nota, trazamos el camino desde las ecuaciones diferenciales de estado hasta la arquitectura **Mamba**, que está redefiniendo los límites de la modelización de secuencias.

---

## 1. De RNN a SSM

Las redes neuronales recurrentes (RNN) clásicas procesan una secuencia $x(t)$ manteniendo un estado oculto $h(t)$ que se actualiza secuencialmente:

$$
h_t = \sigma(W_h h_{t-1} + W_x x_t + b)
$$

Aunque tienen complejidad lineal en la longitud de secuencia y memoria constante $O(1)$ por paso, sufren de gradientes que desaparecen/explotan y son inherentemente secuenciales, lo que las hace lentas de entrenar.

Los **State Space Models** se inspiran en la teoría de control lineal. En tiempo continuo, un SSM se define por dos ecuaciones:

$$
\dot{h}(t) = A h(t) + B x(t)
$$

$$
y(t) = C h(t) + D x(t)
$$

Donde:
- $A \in \mathbb{R}^{N \times N}$ es la matriz de transición de estado.
- $B \in \mathbb{R}^{N \times 1}$ es la matriz de entrada.
- $C \in \mathbb{R}^{1 \times N}$ es la matriz de salida.
- $D \in \mathbb{R}^{1 \times 1}$ es un término de skip-connection.

Para aplicarlo a secuencias discretas (tokens), se aplica una **discretización** con paso $\Delta$ (equivalente al intervalo de muestreo). Utilizando la transformación bilineal (Tustin) o método de Euler:

$$
\bar{A} = (I - \Delta/2 \cdot A)^{-1} (I + \Delta/2 \cdot A)
$$

$$
\bar{B} = (I - \Delta/2 \cdot A)^{-1} \Delta B
$$

Las ecuaciones discretas resultantes son:

$$
h_k = \bar{A} h_{k-1} + \bar{B} x_k
$$

$$
y_k = C h_k + D x_k
$$

Esta forma recuerda a una RNN, pero con una estructura lineal (sin no-linealidad punto a punto en la recurrencia), lo que permite desenrollarla como una convolución.

---

## 2. HiPPO Theory

El desafío fundamental de un SSM es que la matriz $A$ debe estar diseñada para que el estado $h(t)$ comprima de manera efectiva la historia de la entrada $x(\tau)$ para $\tau < t$. **HiPPO** (High-order Polynomial Projection Operators) resuelve esto mediante teoría de aproximación funcional.

La idea de HiPPO es proyectar la historia de la señal de entrada sobre una base de polinomios ortogonales (ej. Legendre) de forma online. La matriz $A$ se deriva para que el estado $h(t)$ codifique los coeficientes de esta proyección.

Para la medida de tiempo continuo **LegS** (Scaled Legendre), la matriz HiPPO tiene entradas:

$$
A_{nk} = -\frac{1}{2} \cdot (2n+1)^{1/2} (2k+1)^{1/2} \cdot \mathbb{1}_{n > k}
$$

con una estructura particular que garantiza que el estado olvide información antigua de manera exponencial mientras retiene información reciente con mayor fidelidad.

Matemáticamente, esto se conecta con la teoría de memoria de largo plazo: el kernel de convolución implícito de un SSM con HiPPO decae de forma controlada, permitiendo capturar dependencias a distancias arbitrarias sin perder estabilidad numérica.

Caso real: Los primeros experimentos de HiPPO en tareas de MNIST serializada (784 pixels como secuencia) demostraron que SSM con inicialización HiPPO superaban a LSTM estándar con una fracción de los parámetros.

💡 **Tip:** En la práctica, no es necesario derivar $A$ manualmente. Las librerías como `mamba-ssm` inicializan $A$ con las fórmulas HiPPO cerradas. Lo importante es entender que $A$ no es aleatoria: está diseñada para ser una memoria optimizada.

---

## 3. S4 (Structured State Space for Sequence Modeling)

**S4** (Gu et al., 2021) resuelve el problema computacional del SSM. Aunque la forma recurrente es lineal en $L$, entrenarla secuencialmente es inaceptablemente lento. S4 observa que si $A$ es **diagonalizable**, la recurrencia puede expresarse como una **convolución**:

$$
y = \bar{K} * x
$$

donde el kernel $\bar{K}$ es la respuesta al impulso del sistema. Para una matriz $A$ diagonalizable como $A = V \Lambda V^{-1}$, los elementos del kernel pueden calcularse eficientemente en el dominio de Fourier mediante la **Transformada Rápida de Fourier (FFT)**:

$$
\bar{K} = \mathcal{F}^{-1}\left( \mathcal{F}(C e^{\bar{A} t} \bar{B}) \right)
$$

Esto reduce la complejidad de entrenamiento de $O(L^2 N)$ a $O(L \log L)$, comparable a la FFT, y permite paralelización masiva en GPU.

S4 impone una estructura normal plus low-rank (NPLR) a $A$, que garantiza estabilidad numérica y diagonalización eficiente. La complejidad total de S4 es:

$$
\text{Entrenamiento: } O(B L \log L) \quad \text{Inferencia: } O(B L N)
$$

donde $B$ es batch size, $L$ es longitud de secuencia, y $N$ es el tamaño del estado (típicamente $N=64$).

⚠️ **Advertencia:** Aunque S4 es lineal en $L$, la constante $N$ y los overhead de FFT pueden hacer que para secuencias cortas (< 1k tokens) un Transformer optimizado con FlashAttention sea más rápido. S4 brilla en $L > 4k$.

---

## 4. Diagonalization y Cálculo Eficiente

La clave del rendimiento de S4 radica en no materializar la matriz completa $A$. En su lugar, opera sobre su representación diagonal y componentes de bajo rango.

Dada la descomposición $A = V \Lambda V^{-1}$, el cálculo del estado se factoriza:

$$
h_k = V \Lambda^k V^{-1} h_0 + \sum_{j=0}^{k-1} V \Lambda^{k-1-j} V^{-1} \bar{B} x_j
$$

El término $V^{-1} \bar{B}$ y $C V$ se precomputan. La recurrencia sobre la diagonal $\Lambda$ se reduce a productos elemento a elemento (Hadamard), que son altamente paralelizables.

En la práctica, S4 utiliza una aproximación donde $A$ es la suma de una matriz normal y una perturbación de rango 1, permitiendo el uso de algoritmos de Cauchy para el cálculo del kernel en $O(N L)$ en lugar de $O(N^2 L)$.

---

## 5. Mamba (Selective State Space)

**Mamba** (Gu & Dao, 2023) identifica la limitación principal de S4: las matrices $B$, $C$ y el paso de discretización $\Delta$ son **globales** (compartidas para todos los tokens en una capa). Esto significa que el modelo no puede ignorar información irrelevante ni enfocarse en detalles específicos del token actual, una capacidad inherente a la auto-atención.

Mamba hace que $B$, $C$ y $\Delta$ sean **funciones de la entrada**:

$$
\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta)
$$

$$
B_t = W_B x_t
$$

$$
C_t = W_C x_t
$$

Esto convierte al SSM en un sistema **selectivo** y dependiente del tiempo, donde cada token tiene su propia dinámica de transición de estado.

### Parallel Associative Scan

El problema computacional es que la recurrencia ya no es una convolución fija (porque $\bar{A}, \bar{B}$ varían con $t$), por lo que la FFT no aplica. Mamba resuelve esto mediante un **scan asociativo paralelo**, que computa la recurrencia en $O(L)$ pasos utilizando $O(\log L)$ profundidad de paralelización:

Dada la recurrencia $h_k = \bar{A}_k h_{k-1} + \bar{B}_k x_k$, definimos la operación asociativa sobre tuplas $(a, b)$:

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)
$$

El scan paralelo aplica esta operación en árbol sobre la secuencia, logrando entrenamiento eficiente en GPU.

La complejidad final de Mamba es:

$$
\text{Entrenamiento: } O(B L D N) \quad \text{Inferencia: } O(B D N) \text{ por paso}
$$

Donde $D$ es la dimensión del modelo y $N$ es el estado SSM (típicamente 16 o 64). Crucialmente, la inferencia es **constante en $L$** por paso, como una RNN, pero con la calidad de un Transformer.

---

## 6. Comparativa Transformer vs Mamba

| Característica | Transformer | Mamba |
|----------------|-------------|-------|
| Complejidad entrenamiento | $O(L^2 D)$ | $O(L D N)$ |
| Complejidad inferencia (por paso) | $O(L D)$ (KV cache lineal) | $O(D N)$ (constante en L) |
| Memoria inferencia | $O(L D)$ | $O(D N)$ |
| Paralelización en entrenamiento | Completa | Completa (scan) |
| Selectividad (input-dependent) | Sí (atención) | Sí (selective SSM) |
| Comunicación multi-GPU | Alta (all-reduce) | Media |

La ventaja de Mamba en inferencia es abrumadora para secuencias largas. Mientras que un Transformer debe cargar un KV cache de tamaño $O(L)$, Mamba solo necesita mantener el estado $h_t$ de tamaño $O(N)$.

Para $L = 100k$ tokens, $D = 4096$, $N = 64$:

$$
M_{KV}^{Transformer} \propto 100k \times 4096 \approx 409.6 \, \text{M valores}
$$

$$
M_{State}^{Mamba} \propto 64 \times 4096 \approx 262 \, \text{k valores}
$$

Una reducción de tres órdenes de magnitud en memoria de estado.

Caso real: **Mamba** fue evaluado en el benchmark de long-contexto **Long Range Arena (LRA)**, superando a todos los Transformers eficientes (Longformer, Linear Attention, Performer) en tareas de audio, texto largo y secuencias estructuradas.

⚠️ **Advertencia:** Mamba es altamente sensible a la inicialización y al learning rate. Los autores recomiendan learning rates más bajos que los típicos para Transformers (ej. $10^{-3}$ vs $3 \times 10^{-4}$) y un scheduler coseno con warmup prolongado.

---

## 7. Casos de Uso

### Genómica

Las secuencias de ADN humano tienen ~3 mil millares de pares de bases. Los Transformers no pueden atender contextos de esta magnitud. **HyenaDNA** (Poli et al., 2023) y **Caduceus** (Schiff et al., 2024) adaptan arquitecturas SSM para modelar el genoma completo a resolución de nucleótido, logrando predecir regiones regulatorias y efectos de variantes genéticas con contextos de 1M tokens.

### Audio

Un minuto de audio a 16kHz contiene 960k muestras. Los SSM han demostrado ser efectivos en síntesis de voz, separación de fuentes y clasificación de música, donde la dependencia a largo plazo es esencial (estructura rítmica, armonía).

### Documentos y Video

Para análisis de documentos legales o médicos que exceden 100k tokens, o para entender relaciones temporales en video de larga duración, Mamba ofrece una alternativa viable sin los costos prohibitivos de la auto-atención cuadrática.

Caso real: **Together AI** experimentó con modelos híbridos Transformer-Mamba, utilizando capas Transformer en las primeras capas (para capturar dependencias locales finas) y capas Mamba en las capas profundas (para contexto global), logrando un compromiso óptimo entre calidad y eficiencia.

---

## 📦 Código de Compresión: Bloque Mamba Simplificado en PyTorch

El siguiente script proporciona una implementación didáctica de un bloque selective SSM. En producción, se recomienda la implementación CUDA optimizada de `mamba-ssm`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model=512, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # Proyección de entrada
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolución causal local
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Parámetros SSM
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Proyección de salida
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x):
        b, l, d = x.shape
        
        # Proyección y split
        x_and_res = self.in_proj(x)  # [B, L, 2*D_inner]
        x_inner, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Convolución causal
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Parámetros selectivos SSM
        ssm_params = self.x_proj(x_conv)  # [B, L, 2*state + 1]
        B, C, dt = torch.split(ssm_params, [self.d_state, self.d_state, 1], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)).squeeze(-1)  # [B, L, D_inner]
        A = -torch.exp(self.A_log.float())  # [D_inner, state]
        
        # Discretización (Euler)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))  # [B, L, D_inner, state]
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D_inner, state]
        
        # Recurrencia SSM (secuencial para claridad; en producción usar scan paralelo)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y = torch.sum(h * C[:, t].unsqueeze(2), dim=-1)  # [B, D_inner]
            ys.append(y)
        y = torch.stack(ys, dim=1)  # [B, L, D_inner]
        
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(res)
        
        return self.out_proj(y)

# Demo
block = MambaBlock(d_model=512, d_state=64)
x = torch.randn(2, 128, 512)
out = block(x)
print(f"Entrada: {x.shape}, Salida: {out.shape}")
```

---

## 🎯 Proyecto: Benchmark de Longitud de Secuencia Transformer vs SSM

Implementa un experimento controlado que compare un mini-transformer (2 capas, atención multi-cabeza) con un modelo equivalente basado en bloques Mamba en secuencias de longitud variable.

### Configuraciones a Evaluar

| Longitud $L$ | 512 | 2048 | 8192 | 32768 |
|--------------|-----|------|------|-------|
| Transformer (FlashAttention) | ✓ | ✓ | ✓ | ✗ (OOM?) |
| Mamba | ✓ | ✓ | ✓ | ✓ |

### Métricas

1. **Tiempo de entrenamiento por epoch** (forward + backward).
2. **Memoria GPU pico** medida con `torch.cuda.max_memory_allocated`.
3. **Perplexity** en una tarea sintética de copia de secuencia (sequence copying) o en un dataset de texto real (ej. PG-19 para long context).

### Entregables

- Script `benchmark_sequence_length.py`.
- Gráficas de tiempo y memoria vs. longitud de secuencia (escala log-log).
- Análisis del punto de cruce donde Mamba supera al Transformer en eficiencia.

💡 **Tip:** Para secuencias de 32k+, asegúrate de que tu entorno tiene al menos 24 GB de VRAM. Si no es así, utiliza `torch.cuda.amp` (autocast) y gradient checkpointing para el Transformer, y compara contra Mamba en FP16.

⚠️ **Advertencia:** La implementación naive de la recurrencia SSM en PyTorch puro (como la mostrada arriba) es extremadamente lenta para entrenamiento. Para el benchmark de larga secuencia, utiliza la librería oficial `mamba-ssm` con kernels CUDA optimizados, o limita el tamaño del modelo para que el experimento sea manejable en tiempo razonable.

---

![Estructura del ADN](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/DNA_double_helix_horizontal.png/640px-DNA_double_helix_horizontal.png)

---

**Enlaces internos:**
- [[00 - Bienvenida]]
- [[01 - Mixture of Experts]]
- [[03 - Modelos Multimodales de LLMs]]
- [[04 - Caso Practico - Agente Autonomo con LLM]]
