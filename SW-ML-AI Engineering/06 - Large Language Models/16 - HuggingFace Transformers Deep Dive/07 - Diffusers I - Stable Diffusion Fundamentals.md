# 🎨 Diffusers I - Stable Diffusion Fundamentals

## 🎯 Learning Objectives

- Understand the latent diffusion architecture: VAE, UNet, and CLIP text encoder
- Master the `diffusers` library core classes: `Pipeline`, `Scheduler`, `UNet2DConditionModel`
- Configure inference parameters: `guidance_scale`, `num_inference_steps`, `negative_prompt`
- Use different schedulers: DDPM, PNDM, DPM++ 2M Karras, Euler
- Run image-to-image translation and inpainting with specialized pipelines
- Connect diffusion theory to the practical `StableDiffusionPipeline` API
- Analyze how noise transforms into coherent images step-by-step through the reverse process

---

## Introduction

Generative AI has captured the world's imagination, and at the heart of this revolution lies latent diffusion. While [[06 - Large Language Models]] taught us to generate text token-by-token, diffusion models generate images latent-by-latent. Stable Diffusion, released by Stability AI in 2022, democratized high-quality image generation by operating in latent space rather than pixel space, reducing memory requirements from A100-only (16+ GB) to consumer GPUs (6–8 GB).

The `diffusers` library is HuggingFace's dedicated toolkit for diffusion models. It decouples the generation process into four modular components: a **pipeline** orchestrates the loop, a **scheduler** defines the noise update rule, a **UNet** predicts noise, and a **VAE** compresses and decompresses images between pixel and latent space. This modularity allows engineers to swap schedulers (DPM++ for speed, Euler for creativity), fine-tune UNets for specific styles (DreamBooth), and compose pipelines (img2img, inpainting, ControlNet) without rewriting the inference stack.

This note is foundational for [[08 - Diffusers II - Advanced Pipelines and ControlNet]], where we extend these basics with ControlNet conditioning, LoRA adaptation, and community pipelines. Understanding the fundamentals here is also essential for [[09 - MLOps y Produccion]] when deploying diffusion endpoints that must handle concurrent requests within strict latency budgets.

---

## 1. Latent Diffusion Architecture

### The Forward and Reverse Process

Diffusion models learn to reverse a gradual noising process. Given a clean image $x_0$, a forward process adds Gaussian noise over $T$ timesteps according to a variance schedule $\beta_1, \beta_2, \ldots, \beta_T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I)$$

Because the sum of Gaussian distributions is itself Gaussian, this process can be expressed in closed form at any timestep $t$:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon \quad \text{where}\ \epsilon \sim \mathcal{N}(0, I)$$

and $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$. As $t \to T$, $\bar{\alpha}_t \to 0$ and $x_t$ approaches pure Gaussian noise $\mathcal{N}(0, I)$.

The reverse process learns to undo this noising. A neural network $\epsilon_\theta$ (the UNet) is trained to predict the noise $\epsilon$ that was added at each timestep $t$:

$$\mathcal{L} = \mathbb{E}_{t \sim [1,T],\ x_0,\ \epsilon \sim \mathcal{N}(0,I)} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

At inference, we start from pure noise $x_T \sim \mathcal{N}(0, I)$ and iterate:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

where $\alpha_t = 1 - \beta_t$, $\sigma_t$ controls stochasticity, and $z \sim \mathcal{N}(0, I)$. For text conditioning, the UNet also receives the text embedding $c$, giving $\epsilon_\theta(x_t, t, c)$.

### The VAE: Compressing to Latent Space

Stable Diffusion's critical innovation is operating in latent space. Instead of diffusing over $512 \times 512 \times 3$ pixel tensors (~786k values), a Variational Autoencoder (VAE) compresses images into a $64 \times 64 \times 4$ latent tensor (~16k values)—a **49x reduction**. The VAE consists of an encoder $E$ and decoder $D$:

$$z = E(x), \quad \hat{x} = D(z)$$

The VAE is trained with a perceptual + L1 reconstruction loss plus a small KL penalty to keep the latent distribution close to a standard normal:

$$\mathcal{L}_{\text{VAE}} = \|x - D(E(x))\|_1 + \lambda_{\text{perceptual}} \cdot \text{LPIPS}(x, \hat{x}) + \beta \cdot \text{KL}(E(x) \mid \mathcal{N}(0, I))$$

The KL penalty is intentionally weighted very low ($\beta \approx 10^{-6}$) to avoid distorting the reconstruction quality while allowing some regularity in the latent space for diffusion.

### The UNet: Predicting Noise in Latent Space

The UNet2DConditionModel is a U-shaped architecture with downsampling and upsampling paths connected by skip connections. Each block contains:

- **ResNet blocks** with group normalization and SiLU activations
- **Spatial self-attention** layers that relate features across spatial positions
- **Cross-attention** layers where the query comes from the spatial features and the key/value come from the CLIP text embedding, injecting text conditioning:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q_{\text{spatial}} K_{\text{text}}^T}{\sqrt{d}}\right) V_{\text{text}}$$

The time step $t$ is encoded via sinusoidal positional encoding (same as Transformer) and injected into each ResNet block via scale-and-shift (FiLM) modulation.

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse, high quality, 8k"
negative_prompt = "blurry, low quality, deformed"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("astronaut.png")
```

### The Noise Schedule

The noise schedule $\beta_1, \ldots, \beta_T$ determines how quickly information is destroyed during the forward process and, consequently, how the reverse process reconstructs it. A linear schedule (used in DDPM) adds noise evenly across timesteps, but this wastes capacity—most of the structural information is destroyed early, and later timesteps add imperceptible noise.

Stable Diffusion uses a **scaled linear** or **cosine** schedule (cosine is smoother and preserves more information for longer). The cosine schedule defines $\bar{\alpha}_t$ directly:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

where $s \approx 0.008$ prevents $\beta_t$ from being too small at $t=0$. The cosine schedule ensures a gradual information destruction curve, making the reverse process's job more uniform across timesteps.

### Classifier-Free Guidance (CFG)

CFG is essential for prompt adherence. During training, the text conditioning $c$ is randomly dropped with probability $p_{\text{drop}} \approx 10\%$ and replaced with a null embedding $\emptyset$. At inference, the predicted noise is extrapolated away from the unconditioned prediction toward the conditioned one:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

where $w$ is the guidance scale. With $w = 1$, the model follows its conditioned prediction. With $w > 1$, the extrapolation amplifies text conditioning. Values above 15 can cause over-saturation because the linear extrapolation assumption breaks down at extreme distances from the unconditioned manifold.

❌ **Antipattern**: Setting `guidance_scale > 15` without increasing `num_inference_steps`. The noise extrapolation overshoots, producing repetitive patterns and out-of-gamut colors.

✅ **Correct**: Use the balanced standard: `guidance_scale=7.5` with `num_inference_steps=50`. For faster generation, reduce both proportionally (e.g., 7.5 with 25 DPM++ steps).

💡 **Tip**: Always set `torch_dtype=torch.float16` on GPU for a ~2x speedup and ~50% memory reduction. On CPU, use FP32—FP16 is unsupported on CPU and produces black images or NaNs.

### Training Diffusion Models

Training a diffusion model from scratch requires thousands of GPU-hours, but understanding the training loop informs production decisions about fine-tuning and data curation. The training loop for each batch:

1. Sample clean images $x_0$ from the training distribution
2. Sample timesteps $t \sim \text{Uniform}[1, T]$
3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$ and create noisy images $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
4. Compute noise prediction $\hat{\epsilon} = \epsilon_\theta(x_t, t, c)$ where $c$ is the text embedding (with $p_{\text{drop}}$ probability of being null for CFG)
5. Minimize $\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$

The loss is simple MSE, but the model must learn widely different behaviors at different timesteps—at small $t$ it removes fine-grained pixel noise, at large $t$ it hallucinates global structure. This makes the UNet's time embedding crucial: it modulates the behavior of every ResNet block via FiLM modulation as a function of $t$.

For fine-tuning (DreamBooth, LoRA), the base model is already converged, so only 500–2000 steps with a very low learning rate ($10^{-5}$) are needed on 3–20 curated images.

![Diffusion Process](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Stable_Diffusion_Diagram.png/800px-Stable_Diffusion_Diagram.png)

**Caso real**: Canva integrated Stable Diffusion into its design platform, allowing 100M+ users to generate images directly in the editor. Their team optimized inference with TensorRT-compiled UNets and custom scheduler tuning on A10G GPUs, achieving sub-2-second generation at 512×512. The critical optimization was moving the VAE decode to a batched post-processing step—decoding one latent at a time underutilizes GPU tensor cores, so they accumulate a batch of latents from concurrent requests and decode them together, improving throughput by 3x.

---

## 2. Schedulers and Inference Parameters

### The Mathematics of Denoising

The scheduler defines the mathematical rule for updating the latent at each timestep using the UNet's noise prediction. Different schedulers solve the reverse-time stochastic differential equation (SDE) or ordinary differential equation (ODE) with varying orders of accuracy and stochasticity.

**DDPM** (Ho et al., 2020) follows the full reverse SDE with stochasticity:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

where $\sigma_t^2 = \beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$. The stochastic term $z$ adds noise at each step, which can improve sample diversity but requires many steps (1000) because each step removes only a small amount of noise.

**DDIM** (Song et al., 2021) removes the stochastic term, making the process a deterministic ODE:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

This allows skipping steps: we can sample every $k$-th timestep and still produce valid samples because the ODE defines a unique trajectory. DDIM typically uses 50 steps.

**DPM++ 2M Karras** uses a second-order solver. Instead of updating based on the current gradient evaluation alone, it uses a linear combination of the current and previous evaluations (like Heun's method for ODEs):

$$x_{t-1} = \frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} x_t + \left( \frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} - 1 \right) \cdot \left( \frac{3}{2}\epsilon_\theta(x_t, t) - \frac{1}{2}\epsilon_\theta(x_{t+1}, t+1) \right)$$

The "2M" means it uses multistep (two previous noise predictions), and "Karras" refers to the noise schedule from Karras et al. (2022) which concentrates more steps near the end of denoising where fine details emerge. DPM++ achieves high quality in just 20–25 steps.

**Euler a** (ancestral) adds stochasticity back with a different formulation. It uses the Euler method for the ODE solver but adds noise at each step scaled by the current sigma:

$$x_{t-1} = x_t + (t - t') \cdot \epsilon_\theta(x_t, t) + \mathcal{N}(0, \sigma_t^2)$$

This stochasticity can produce more creative textures and reduce repetitive patterns, making it popular for artistic applications.

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe(
    "a cyberpunk cityscape at night, neon lights, rain",
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(123)
).images[0]
```

❌ **Antipattern**: Changing the scheduler without calling `from_config()`. Each scheduler has different configuration keys (`trained_betas`, `prediction_type`, `steps_offset`). A mismatched config causes incorrect step calculations, producing black images.

✅ **Correct**: Always use `SchedulerClass.from_config(pipe.scheduler.config)` to inherit the correct noise schedule parameters from the checkpoint.

❌ **Antipattern**: Using `num_inference_steps < 10` with DDPM or DDIM. The step size becomes too large—the linear approximation of the denoising trajectory breaks down, and the output is indistinguishable from noise.

✅ **Correct**: Use DPM++ for low-step regimes (15–25 steps). For ultra-fast generation (< 10 steps), use a consistency model or LCM scheduler.

| Scheduler | Steps | Quality | Deterministic | Best For |
|-----------|-------|---------|---------------|----------|
| DDPM | 1000 | Reference | No | Training, not inference |
| DDIM | 50 | Good | Yes | Deterministic, consistent seeds |
| PNDM | 50 | Good | Yes | Stable, pseudo-numerical |
| DPM++ 2M Karras | 20–25 | Excellent | Yes | General purpose, best quality/speed |
| Euler a | 30 | Good | No | Artistic, creative textures |
| DPM++ 2S a | 20 | Good | No | Fast, ancestral (creative) |

![Euler Method](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Euler_method.png/640px-Euler_method.png)

💡 **Tip**: For production APIs, use DPM++ 2M Karras with 25 steps as the default. It provides the best quality-to-speed ratio across all schedule types. Offer Euler a as a "creative mode" option for users who want more variation.

**Caso real**: Midjourney uses proprietary scheduler tuning and custom noise schedules to achieve its distinctive aesthetic in 30–40 steps. Their team discovered that sampling steps should be concentrated in different noise regimes depending on the prompt type: landscape prompts benefit from more steps at high noise levels (global composition), while portrait prompts benefit from more steps at low noise levels (facial detail). This scheduler-level optimization is a key competitive differentiator.

---

## 3. Image-to-Image and Inpainting

### Controlling the Initial Latent

Text-to-image starts from random noise. Image-to-image starts from a noised version of an existing image's latent encoding, steering the generation toward the source while allowing creative transformation.

The `strength` parameter controls this tradeoff. Given source image $x_{\text{src}}$, we encode to latent $z_{\text{src}} = E(x_{\text{src}})$ and compute the starting timestep:

$$t_{\text{start}} = \lfloor \text{strength} \cdot T \rfloor$$

Then we noise the latent:

$$z_{t_{\text{start}}} = \sqrt{\bar{\alpha}_{t_{\text{start}}}}\, z_{\text{src}} + \sqrt{1 - \bar{\alpha}_{t_{\text{start}}}}\, \epsilon$$

At $\text{strength} = 0.0$: $t_{\text{start}} = 0$, no noise is added, and the output equals the input. At $\text{strength} = 1.0$: $t_{\text{start}} = T$, the source is completely replaced by pure noise, and generation is equivalent to text-to-image from scratch.

The effective denoising trajectory is shorter when strength is lower. With strength $s$, the UNet only runs $(1-s) \cdot T$ steps—fewer steps means less transformation and faster inference. A strength of 0.5 with $T=50$ executes only 25 denoising steps, halving the generation time.

### Inpainting with the Mask Channel

Inpainting extends img2img by masking a region to regenerate. The UNet is modified to accept 5-channel input instead of 4: the 4 latent channels are concatenated with a 1-channel mask where white = regenerate and black = preserve.

During denoising, the unmasked region is clamped to the noised source latent at each step:

$$z_t = m \odot z_t^{\text{(denoised)}} + (1 - m) \odot \sqrt{\bar{\alpha}_t}\, z_{\text{src}} + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

where $m$ is the mask (1 for masked region, 0 for preserved). This ensures the unmasked area remains faithful to the source while the masked area is freely generated. The inpainting UNet must be explicitly trained on this 5-channel input format to learn how to blend generated content with preserved context.

```python
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import torch

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("photo.jpg").convert("RGB").resize((512, 512))

result = pipe_i2i(
    prompt="turn this into a cyberpunk scene, neon lights, rain",
    image=init_image,
    strength=0.75,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

mask = Image.open("mask.png").convert("L").resize((512, 512))

result = pipe_inpaint(
    prompt="a cute cat sitting on the bench",
    image=init_image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

❌ **Antipattern**: Using a standard text-to-image UNet (`runwayml/stable-diffusion-v1-5`) for inpainting. The standard UNet expects 4-channel latents; inpainting requires 5 channels (latent + mask). This causes a `RuntimeError: shape mismatch` at the first convolutional layer.

✅ **Correct**: Use `runwayml/stable-diffusion-inpainting` or any checkpoint whose UNet has `in_channels=5`. Check this via `pipe.unet.config.in_channels`.

❌ **Antipattern**: Setting `strength=1.0` in img2img and expecting the composition to match the source. The original image is completely overwritten—you get a text-to-image result with no connection to the input.

✅ **Correct**: Start with `strength=0.5`. For preserving pose and composition, use 0.3–0.4. For strong creative transformation while keeping color palette, use 0.6–0.7. Always preview at multiple strengths.

![Inpainting Example](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Image_inpainting_example.png/640px-Image_inpainting_example.png)

💡 **Tip**: For batch img2img workflows, compute the VAE encoding once outside the loop. Precompute `latents = pipe.vae.encode(pipe.image_processor(init_image).unsqueeze(0).to("cuda"))` and pass the latent directly to multiple generation calls with different prompts or strengths. This can reduce per-image latency by 15–20%.

**Caso real**: Adobe Photoshop's Generative Fill uses a specialized inpainting diffusion model fine-tuned for high-resolution, semantically aware object replacement. The mask is generated interactively as the user paints with a brush tool. The critical engineering challenge was achieving sub-second latency for interactive use—they solved it with a distilled UNet that runs only 10 denoising steps using a consistency-based scheduler, trading 5% quality for 5x speed. Users perceive the result as nearly instant, making the feature usable in a real-time editing workflow.

---

## 📦 Compression Code

```python
"""Stable diffusion fundamentals: text2img, img2img, inpainting, scheduler swap."""
import torch
from diffusers import (
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler,
)
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
img = pipe("a serene lake at sunrise, oil painting", num_inference_steps=25,
           guidance_scale=7.5, generator=torch.Generator("cuda").manual_seed(42)).images[0]
img.save("text2img.png")

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
src = Image.open("text2img.png").convert("RGB").resize((512, 512))
img2 = pipe_i2i(prompt="cyberpunk version", image=src, strength=0.6, num_inference_steps=30).images[0]
img2.save("img2img.png")

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")
mask = Image.new("L", (512, 512), 0)
mask.paste(255, (200, 200, 300, 300))
img3 = pipe_inpaint(prompt="a glowing orb", image=src, mask_image=mask, num_inference_steps=30).images[0]
img3.save("inpaint.png")
```

## 🎯 Key Takeaways

- Stable Diffusion operates in latent space via a VAE that compresses 512×512×3 pixels into 64×64×4 latents, a 49x reduction enabling consumer GPU inference.
- The UNet predicts noise conditioned on CLIP text embeddings via cross-attention; the scheduler defines the discrete update rule from noise to image.
- Classifier-free guidance extrapolates between conditioned and unconditioned noise predictions; 7.5 is the standard default scale.
- Schedulers determine the step count vs quality tradeoff: DPM++ 2M Karras achieves excellent quality in 20–25 steps, outperforming DDPM's 1000-step requirement.
- Image-to-image uses `strength` (0–1) to control source structure preservation—lower values preserve, higher values allow creativity.
- Inpainting requires a 5-channel UNet (4 latent + 1 mask) to blend generated content with preserved context seamlessly.
- Always use FP16 on GPU, match scheduler configs to the checkpoint via `from_config`, and benchmark multiple schedulers for your specific quality and latency requirements.

## References

- Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR.
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
- Song et al. (2021). "Denoising Diffusion Implicit Models." ICLR.
- Lu et al. (2022). "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models." NeurIPS.
- Karras et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS.
- HuggingFace Diffusers Documentation: https://huggingface.co/docs/diffusers
- Stable Diffusion v1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
