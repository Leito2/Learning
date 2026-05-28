# 🏷️ Hybrid Precision — FP8 Quantization and Transformer Engines

## 🎯 Learning Objectives
- Understand why INT8 fails catastrophically for LLM activations while FP8 succeeds
- Distinguish E4M3 (weight-optimal) vs E5M2 (activation-optimal) FP8 formats
- Implement per-tensor scaling and calibration for FP8 quantization
- Leverage H100/B200 Transformer Engine hardware for 2× throughput on FP8 GEMM
- Compare FP8 against FP16, BF16, and INT8 across accuracy, throughput, and hardware support

## Introduction

**FP16/BF16 inference wastes 8 bits per value on precision the model doesn't need. FP8 halves VRAM and doubles throughput — and unlike INT8, it doesn't destroy accuracy.** This is the most impactful, lowest-effort optimization in the LLM deployment stack. It's natively supported by H100/B200 hardware, requires zero model retraining (just calibration), and delivers consistent 1.8-2× throughput improvements. If you're not shipping FP8 quantized models in production, you're leaving free GPU capacity on the table.

The name "hybrid precision" captures the crucial insight: ONE format does NOT fit all tensors. Weights and activations have fundamentally different statistical properties. Weights are relatively well-behaved (Gaussian-like distributions centered near zero with moderate variance). Activations have heavy-tailed distributions with large outliers — a single neuron can fire 100× harder than average on certain inputs. Using the same format for both guarantees suboptimal results. The solution uses two FP8 variants: **E4M3 for weights** (prioritizes precision in the range where most weights live) and **E5M2 for activations** (prioritizes dynamic range to capture outlier activations without clipping).

Before FP8, the quantization landscape was dominated by INT8. INT8 works well for weights (uniform quantization bins, weights are somewhat uniform) but catastrophically for activations. The outlier problem: if 99.9% of activation values lie in [-2, 2] but one outlier is 500, uniform INT8 quantization must allocate half its bins to [-500, -2] and [2, 500] — leaving only 128/2 = 64 levels for the 99.9% of values in [-2, 2]. The effective precision in the dense region drops to approximately $4/64 \approx 0.06$ per bin — barely 4 bits of effective precision. FP8 solves this with its floating-point format: a single exponent bit difference handles the dynamic range, while mantissa bits provide uniform precision within each exponent range. This matters for [[06/09 - Sistemas de LLMs en Producción]] because accuracy degradation in production is measured in user-facing metrics (wrong answers, safety violations), not abstract perplexity scores.

---

## 1. Why INT8 Fails for LLMs

To understand FP8's success, you must first understand INT8's failure mode. INT8 represents numbers in the range $[-128, 127]$ (signed) or $[0, 255]$ (unsigned) with uniform spacing of 1.0. The quantization mapping is:

$$x_{\text{int8}} = \text{round}\left(\frac{x_{\text{fp16}} - z}{s}\right)$$

where $s$ is the scale factor (the step size) and $z$ is the zero point offset. The quantization error for a value $x$ is bounded by:

$$|x - \hat{x}| \leq \frac{s}{2}$$

The problem: $s$ is a **single scalar for the entire tensor** (per-tensor quantization) or a vector (per-channel). The quantization error is uniform — ±s/2 for every value, regardless of magnitude. For a value at 500, an error of ±0.5 (s=1) is 0.1% relative error — excellent. For a value at 0.01, an error of ±0.5 is 5000% relative error — catastrophic. The relative error is:

$$\epsilon_{\text{rel}} = \frac{|x - \hat{x}|}{|x|} \leq \frac{s}{2|x|}$$

As $x \to 0$, $\epsilon_{\text{rel}} \to \infty$. This is the fundamental weakness of fixed-point/INT8: it has constant absolute error, not constant relative error. Deep neural networks, especially transformers, have large quantities of small-magnitude values (near-zero activations after LayerNorm, attention scores for irrelevant tokens) that INT8 destroys.

**Empirical evidence:** When quantizing Llama-7B to INT8 (weights + activations), perplexity on WikiText-2 degrades from 5.68 (FP16) to 6.42 (INT8 W+A) — a 13% increase. On MMLU benchmark, accuracy drops from 45.3% to 38.1% — a 7.2 percentage point loss. This is not acceptable for production.

**The outlier channel problem:** In transformer activations, certain feature dimensions consistently produce 10-100× larger values. The distribution within a tensor looks like: most channels [−2, 2], but channels 127, 459, 1023 consistently in [−200, 200]. Per-tensor quantization is dominated by these outlier channels — the scale $s$ is set by the maximum magnitude, which comes from outlier channels. Per-channel quantization solves this for weights (which are static) but adds runtime overhead for activations (which change per input).

![Precision comparison between FP32 and FP16](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Float_example.svg/640px-Float_example.svg.png)

## 2. FP8: Floating Point to the Rescue

FP8 retains the floating-point structure: a sign bit, exponent bits, and mantissa (fraction) bits. The value represented is:

$$x = (-1)^S \times 2^{E - \text{bias}} \times (1 + M)$$

where $S$ is the sign, $E$ is the exponent (unsigned integer), bias is the exponent offset, and $M$ is the fractional mantissa part (a value in $[0, 1)$ with $M = \sum_{i=1}^{b_m} m_i \cdot 2^{-i}$).

Unlike INT8 (constant absolute precision), FP8 provides **constant RELATIVE precision** within each exponent range. The relative error is bounded by:

$$\epsilon_{\text{rel}} \leq 2^{-(b_m + 1)}$$ where $b_m$ is the number of mantissa bits. For E4M3 ($b_m = 3$): $\epsilon_{\text{rel}} \leq 2^{-4} = 6.25\%$. For E5M2 ($b_m = 2$): $\epsilon_{\text{rel}} \leq 2^{-3} = 12.5\%$. This holds for ALL magnitudes — no degradation for small values.

### E4M3 (4 exponent bits, 3 mantissa bits)

**Format details:**
- 1 sign bit + 4 exponent bits + 3 mantissa bits = 8 bits
- Exponent bias: 7 (so $E_{\text{actual}} = E - 7$)
- Range: $[-240, 240]$ (maximum representable: $2^{8-7} \times (1 + 7/8) = 2^8 \times 1.875 = 240$)
- Minimum positive normal: $2^{-6} = 0.015625$
- Subnormal support: yes (can represent values down to $2^{-9} = 0.00195$ via subnormals)
- Precision: ~3.125% relative error average within range

**Best for: WEIGHTS.** Weights have moderate ranges (typically ±0.1 to ±5 after training), no extreme outliers, and benefit from higher precision (3 mantissa bits) for accurate dot product accumulation.

### E5M2 (5 exponent bits, 2 mantissa bits)

**Format details:**
- 1 sign bit + 5 exponent bits + 2 mantissa bits = 8 bits
- Exponent bias: 15
- Range: $[-57344, 57344]$ (maximum: $2^{31-15} \times (1 + 3/4) = 2^{16} \times 1.75 = 57344$)
- Minimum positive normal: $2^{-14} \approx 6.1 \times 10^{-5}$
- Precision: ~6.25% relative error average within range

**Best for: ACTIVATIONS (and gradients in training).** Activations have heavy-tailed distributions with outliers — values like 500+ from certain neurons. The 5 exponent bits give the dynamic range to represent these without clipping, at the cost of slightly lower precision. But for activations, catching the outlier without clipping is more important than precise representation of small values (which would be clipped to zero in INT8).

💡 E5M2 is also excellent for gradients during FP8 training. Gradient distributions are even more extreme than activation distributions — a single layer's gradient can span 10 orders of magnitude. The 5-bit exponent captures this dynamic range.

**FP8 special values (both formats):**
- NaN: All exponent bits = 1, any non-zero mantissa
- Infinity: All exponent bits = 1, all mantissa bits = 0
- Zero: All exponent bits = 0, all mantissa bits = 0
- Subnormals: All exponent bits = 0, non-zero mantissa ($x = (-1)^S \times 2^{-6} \times M$ for E4M3)

⚠️ FP8 has a narrower range than FP16. When quantizing from FP16/BF16 to FP8, values outside the representable range are clipped, not saturated. This requires proper scaling (next section) to avoid catastrophic accuracy loss from clipping.

![Bit layout comparison](https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/IEEE_754r_Half_Floating_Point_Format.svg/800px-IEEE_754r_Half_Floating_Point_Format.svg.png)

## 3. Per-Tensor Scaling and Calibration

The FP8 formats have limited representable ranges (E4M3: ±240, E5M2: ±57344). To fit FP16 values into these ranges without excessive clipping or underflow, you must SCALE the tensor appropriately.

**Per-tensor scaling:** For each tensor (weight matrix, activation tensor), compute a single scalar scaling factor $s$:

$$x_{\text{fp8}} = Q\left(\frac{x_{\text{fp16}}}{s}\right)$$

where $Q(\cdot)$ rounds to the nearest representable FP8 value. At dequantization (before use):

$$x_{\text{reconstructed}} = x_{\text{fp8}} \cdot s$$

The choice of $s$ is critical. If $s$ is too small, large values clip (saturate the FP8 range). If $s$ is too large, small values underflow (become zero or subnormal). There are several heuristics:

**Max-based scaling:**
$$s_{\max} = \frac{\max(|x_{\text{fp16}}|)}{\text{FP8\_MAX}}$$
Ensures no clipping, but wastes representation range if there's a single outlier much larger than all other values.

**MSE-optimal scaling (per-channel):**
$$s_{\text{opt}} = \arg\min_s \mathbb{E}_x\left[\left(x - Q(x/s) \cdot s\right)^2\right]$$
Minimizes reconstruction error over the tensor distribution. Computationally expensive but can be done offline for static weights.

**Quantile-based scaling:**
$$s_{p} = \frac{\text{quantile}(|x_{\text{fp16}}|, p)}{\text{FP8\_MAX}}$$
Use the $p$-th percentile (e.g., $p=0.999$) instead of the absolute maximum. Trades a tiny amount of clipping (<0.1% of values) for better precision on the bulk of the distribution.

**Calibration process:**
1. Run a small calibration dataset (100-1000 batches) through the model in FP16
2. For each quantizable tensor (weights and activations), collect statistics
3. Compute optimal scaling factors using one of the above methods
4. Store scaling factors alongside the quantized model
5. At inference, apply scaling inline: quantize before GEMM, dequantize after

For weights (static, known at compile time), this is a one-time offline cost. For activations (dynamic, input-dependent), the scaling factor must either be:
- **Static**: Pre-computed from the calibration set (low overhead, slightly suboptimal for out-of-distribution inputs)
- **Dynamic**: Re-computed per batch (higher overhead, better accuracy)

In practice, static scaling factors for activations work well because activation distributions are surprisingly stable across inputs for most layers.

**The GEMM operation in FP8:** The matrix multiplication $Y = A \times B$ becomes:
1. Quantize $A_{\text{fp16}} \to A_{\text{fp8}}$ with scale $s_A$
2. Quantize $B_{\text{fp16}} \to B_{\text{fp8}}$ with scale $s_B$
3. Compute $Y_{\text{fp32}} = A_{\text{fp8}} \times B_{\text{fp8}}$ (accumulate in FP32!)
4. Dequantize: $Y_{\text{fp16}} = Y_{\text{fp32}} \times s_A \times s_B$

⚠️ NEVER accumulate FP8 GEMM results in FP8. The accumulation precision must be at least FP16 (preferably FP32) to avoid catastrophic cancellation when summing many small FP8 values. The H100 Transformer Engine accumulates in FP32 internally.

## 4. H100/B200 Transformer Engine

The NVIDIA Transformer Engine is a library + hardware feature pair that accelerates FP8 computation on Hopper (H100) and Blackwell (B200) architectures. It's not just an INT8 tensor core with FP8 support bolted on — it provides dedicated FP8 GEMM hardware.

**H100 FP8 Performance:**
- FP8 Tensor Core throughput: **1979 TFLOPS** (dense), **3958 TFLOPS** (sparse, 2:4 structured)
- FP16 Tensor Core throughput: **989 TFLOPS** (dense), **1979 TFLOPS** (sparse)
- **Ratio:** FP8 delivers exactly **2× throughput** vs FP16 on the same hardware unit

**B200 FP8 Performance:**
- FP8 Tensor Core throughput: **4500 TFLOPS** (dense), **9000 TFLOPS** (sparse)
- FP16 Tensor Core throughput: **2250 TFLOPS** (dense), **4500 TFLOPS** (sparse)
- Same 2× ratio — FP8 is consistently double the throughput

**Hardware details:** The H100 SM (Streaming Multiprocessor) has 4th-generation Tensor Cores. The FP8 mode operates on 4×4×4 matrix tiles, computing:
$$D = A_{fp8} \times B_{fp8} + C_{fp32}$$

The inputs are FP8 (1 byte each), accumulation is FP32 (4 bytes). This is crucial: the accumulation must preserve numerical precision across many partial sums.

**Pricing advantage:** FP8 reduces memory traffic by 2× (reading 1 byte per value instead of 2). Since LLM inference is memory-bandwidth-bound, this alone provides ~2× speedup even without the 2× compute throughput. It's a rare win-win: both the memory bottleneck AND the compute bottleneck improve.

**NVIDIA NeMo Framework integration:** The Transformer Engine is integrated into NeMo for training. For FP8 training, both forward and backward passes can use FP8 GEMM, but with a critical caveat: the master weights are kept in FP16. Each training step:
1. Quantize master weights FP16 → FP8
2. Forward pass in FP8
3. Backward pass in FP8 (gradients through FP8 GEMM)
4. Dequantize gradients to FP16
5. Update master weights in FP16

This hybrid approach ensures training stability (FP16 weight updates prevent drift from quantization error accumulation) while getting 2× throughput on the expensive GEMM operations.

![NVIDIA H100 GPU](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_Hopper_Architecture.svg/1200px-NVIDIA_Hopper_Architecture.svg.png)

## 5. Production Reality

**Caso real: NVIDIA NeMo Framework ships FP8 training for Llama-3.** NVIDIA's NeMo framework, used for training frontier models like Llama-3, integrates the Transformer Engine for FP8 training. On H100 DGX clusters (8× H100 per node), FP8 training achieves 2× memory reduction and 1.8× throughput improvement over BF16 training. For a 70B model, this means fitting the optimizer states (Adam: 8 bytes per param = 560GB) + gradients (140GB) + activations into fewer GPUs. The 2× memory reduction alone can shift a 64-GPU training run to a 32-GPU run, halving the cluster cost for the same wall-clock time.

**Caso real: Together AI serves FP8-quantized 70B models at 2× the batch size of FP16.** Together AI's inference platform deploys FP8-quantized versions of Llama-3-70B and Mixtral 8×7B. The halved memory footprint allows them to fit 2× the batch size on the same GPU, which doubles tokens-per-second throughput. For their customers, this translates directly to halved cost-per-token — FP8 inference at $0.45/M tokens vs FP16 at $0.90/M tokens. The accuracy difference is <0.1% perplexity increase and zero measurable difference on MMLU.

**Caso real: FP8 deployment at scale on NVIDIA H200.** Cloud providers deploying H200 GPUs (141GB HBM3e, 4.8 TB/s bandwidth) report that FP8 is the DEFAULT precision for LLM inference. The reasoning: H200's 4.8 TB/s bandwidth means the memory wall is pushed further, but FP8 still provides the 2× capacity advantage (fitting 2×-larger models or 2×-longer contexts on the same GPU). No provider deploys FP16 inference on H200 for models >30B parameters — the 141GB is better used with FP8 to serve 70B+ models with long contexts.

**FP8 accuracy results (empirical benchmarks):**

| Model | FP16 PPL | FP8 PPL | ΔPPL | MMLU FP16 | MMLU FP8 | ΔMMLU |
|-------|---------|---------|------|-----------|----------|-------|
| Llama-3-8B | 8.12 | 8.13 | +0.01 | 65.0% | 64.9% | -0.1% |
| Llama-3-70B | 5.84 | 5.85 | +0.01 | 79.5% | 79.4% | -0.1% |
| Mixtral 8×7B | 5.72 | 5.74 | +0.02 | 71.8% | 71.6% | -0.2% |

These results demonstrate that **well-implemented FP8 quantization (E4M3 weights + E5M2 activations) is essentially lossless** on standard benchmarks, while INT8 consistently loses 2-5% absolute on MMLU.

---

## ❌/✅ Comparison: INT8 vs Hybrid FP8

```python
import torch

# ❌ Uniform INT8 quantization of both weights AND activations
def int8_quantize_weights_and_acts(model):
    # Per-tensor symmetric INT8 quantization
    for name, param in model.named_parameters():
        if 'weight' in name:
            max_val = param.abs().max()
            scale = max_val / 127.0
            param.data = (param / scale).round().clamp(-128, 127).to(torch.int8)
            # ⚠️ Activations quantized the same way at runtime — outliers destroy precision
    return model
    # Result: 5-10% accuracy loss on MMLU — unacceptable for production

# ✅ Hybrid FP8: E4M3 for weights, E5M2 for activations
class HybridFP8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.weight_scale = None  # Calibrated offline

    def forward(self, x):
        # 💡 Weights use E4M3 (more mantissa bits = better precision)
        w_fp8 = self.quantize_e4m3(self.weight, self.weight_scale)
        # Activations use E5M2 (more exponent bits = handles outliers)
        x_fp8, x_scale = self.quantize_e5m2_dynamic(x)
        # GEMM in FP32 accumulator
        out = torch._scaled_mm(w_fp8, x_fp8.t(),
                               scale_a=self.weight_scale, scale_b=x_scale,
                               out_dtype=torch.float16)
        return out
    # Result: <0.1% accuracy loss, 2× throughput — production-ready
```

## 6. Code in Practice — FP8 Quantization Simulator

```python
import torch
import numpy as np
from typing import Tuple

def float_to_fp8_e4m3(values: torch.Tensor, scale: float = None) -> Tuple[torch.Tensor, float]:
    """Convert FP32/BF16 → E4M3 (4 exponent, 3 mantissa, bias=7). Best for weights."""
    if scale is None:
        scale = values.abs().max().item() / 240.0  # E4M3 max = 240

    scaled = values / scale
    # Extract sign, exponent, mantissa
    sign = torch.sign(scaled)
    abs_scaled = torch.abs(scaled)

    # Exponent: floor(log2(abs_val))
    exponent = torch.floor(torch.log2(abs_scaled + 1e-12)).clamp(-7, 7).int()

    # Mantissa: 3 bits → 8 quantization levels (0-7)
    mantissa = ((abs_scaled / (2.0 ** exponent.float())) - 1.0) * 8.0
    mantissa = torch.round(mantissa).clamp(0, 7).int()

    # Reconstruct FP8 value
    fp8_int = (sign.sign().int() & 1) + (exponent << 1) + mantissa
    # ¡Sorpresa! FP8 isn't stored as [sign|exp|mant] tightly in real HW —
    # but this captures the correct mathematical mapping

    reconstructed = sign * (2.0 ** exponent.float()) * (1.0 + mantissa.float() / 8.0) * scale
    return reconstructed, scale


def float_to_fp8_e5m2(values: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Convert → E5M2 (5 exponent, 2 mantissa, bias=15). Best for activations."""
    scale = values.abs().max().item() / 57344.0
    scaled = values / scale

    sign = torch.sign(scaled)
    abs_scaled = torch.abs(scaled)
    exponent = torch.floor(torch.log2(abs_scaled + 1e-12)).clamp(-14, 16).int()
    mantissa = ((abs_scaled / (2.0 ** exponent.float())) - 1.0) * 4.0
    mantissa = torch.round(mantissa).clamp(0, 3).int()

    reconstructed = sign * (2.0 ** exponent.float()) * (1.0 + mantissa.float() / 4.0) * scale
    return reconstructed, scale


# Demonstrate the advantage: small values survive in FP8, die in INT8
x = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0])

# INT8 uniform quantization (per-tensor)
int8_max = x.abs().max()  # 500
int8_scale = int8_max / 127.0
x_int8 = torch.round(x / int8_scale).clamp(-128, 127) * int8_scale
print("INT8 small-value loss:", x_int8[:3])
# ⚠️ Output: [0.0, 0.0, 0.0] — small values completely destroyed!

# FP8 E5M2 (activation format with wide dynamic range)
x_fp8, _ = float_to_fp8_e5m2(x)
print("FP8  small-value preservation:", x_fp8[:3])
# 💡 Output: [~0.001, ~0.01, ~0.1] — small values survive!

relative_error = ((x_fp8 - x) / x).abs()
print(f"FP8 mean relative error: {relative_error.mean():.2%}")
```

## 🎯 Key Takeaways
- **INT8 fails for LLM activations because of heavy-tailed outlier distributions** — uniform quantization destroys small values while capturing outliers, losing 5-10% accuracy
- **FP8 solves this with floating-point format: constant RELATIVE error regardless of magnitude** — FP8 E4M3 has ~3.1% relative error; INT8 has catastrophic relative error for small values
- **Hybrid precision uses E4M3 (3-bit mantissa) for weights and E5M2 (5-bit exponent) for activations** — matching format properties to tensor statistical properties
- **FP8 delivers 2× throughput on H100/B200 via dedicated Transformer Engine hardware** — and simultaneously halves memory traffic, doubling effective bandwidth
- **Per-tensor scaling factors must be carefully calibrated** — quantile-based scaling (p=0.999) better than max-based scaling for heavy-tailed activations
- **FP8 GEMM must accumulate in FP32** — accumulation precision loss from FP8 + FP8 → FP8 addition is catastrophic for dot products
- **Empirically, FP8 (E4M3 weights + E5M2 activations) achieves <0.1% perplexity degradation on Llama-70B** — essentially lossless, making it the default production precision

## 🔬 Código de Compresión

```python
"""FP8 quantization simulator: E4M3 for weights, E5M2 for activations in 25 lines."""
import torch, numpy as np

# E4M3 constants
E4M3_MAX, E4M3_BIAS, E4M3_MANT_BITS = 240.0, 7, 3
# E5M2 constants
E5M2_MAX, E5M2_BIAS, E5M2_MANT_BITS = 57344.0, 15, 2

def quant_fp8(x, max_val, bias, mant_bits):
    """Generic FP8 quantizer: sign | exp | mantissa."""
    scale = x.abs().max().item() / max_val
    s, a = torch.sign(x / scale), torch.abs(x / scale)
    exp = torch.floor(torch.log2(a + 1e-12)).clamp(-bias, bias + 1).int()
    n_levels = 2 ** mant_bits
    mant = torch.round(((a / (2.0 ** exp.float())) - 1.0) * n_levels).clamp(0, n_levels - 1).int()
    x_hat = s * (2.0 ** exp.float()) * (1.0 + mant.float() / n_levels) * scale
    return x_hat, scale

def quant_weights_acts(weights, activations):
    """E4M3 (weights) + E5M2 (activations) hybrid quantization."""
    w_q, w_s = quant_fp8(weights, E4M3_MAX, E4M3_BIAS, E4M3_MANT_BITS)
    a_q, a_s = quant_fp8(activations, E5M2_MAX, E5M2_BIAS, E5M2_MANT_BITS)
    # GEMM: accumulate in FP32, then dequantize
    out = torch.mm(a_q.float(), w_q.float().t()) * a_s * w_s
    # ¡Sorpresa! Scaling factors multiply — both quantizations contribute error
    return out.to(torch.float16)  # Dequantize back to FP16

# Demo
w = torch.randn(1024, 1024, dtype=torch.float16)
a = torch.randn(1, 1024, dtype=torch.float16)
a[:, 127] *= 500  # Inject outlier — INT8 would clip this catastrophically
out = quant_weights_acts(w, a)
print(f"Output shape: {out.shape}, mean: {out.mean():.4f}")
```

## References
- Micikevicius, P., et al. (2022). "FP8 Formats for Deep Learning." *arXiv:2209.05433*
- NVIDIA (2023). "Transformer Engine — FP8 Training and Inference." NVIDIA Developer Documentation
- NVIDIA H100 Tensor Core GPU Architecture Whitepaper. *NVIDIA Technical Report, 2022*
- Noune, B., et al. (2022). "8-bit Numerical Formats for Deep Neural Networks." *arXiv:2206.02915*
- Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *arXiv:2208.07339*
- Sun, X., et al. (2024). "FP8-LM: Training FP8 Large Language Models." *arXiv:2310.18313*
- [[06/09 - Sistemas de LLMs en Producción]]
- [[05/03 - Deep Learning con PyTorch]]
- [[05/09 - Deep Learning with TensorFlow]]
