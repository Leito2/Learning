# 🎯 Diffusers II - Advanced Pipelines and ControlNet

## 🎯 Learning Objectives

- Understand ControlNet architecture: duplicated UNet encoder + zero-convolution conditioning
- Implement ControlNet pipelines with canny, depth, pose, and scribble conditioning
- Apply LoRA and load community weights with `load_lora_weights` and merging
- Use IP-Adapter for image prompting and DreamBooth for subject-specific fine-tuning
- Explore Stable Diffusion XL (SDXL) and Flux pipeline differences
- Leverage `diffusers` community pipelines for cutting-edge research implementations
- Design production-ready diffusion systems with conditioning and personalization

---

## Introduction

Basic text-to-image generation is a solved problem; the frontier now lies in controlling what is generated. While [[07 - Diffusers I - Stable Diffusion Fundamentals]] taught us to generate images from noise and text, real-world creative workflows demand precise control over composition, pose, structure, and style. An architect cannot afford to regenerate a building 100 times hoping the perspective aligns—they need a control mechanism.

This note covers the advanced tooling in the `diffusers` ecosystem: ControlNet for structural conditioning, LoRA for efficient style adaptation, IP-Adapter for image-based prompting, and DreamBooth for subject personalization. We also examine SDXL and Flux, the next-generation architectures that push resolution and coherence boundaries. These techniques are essential for ML engineers building generative products in [[09 - MLOps y Produccion]] and creative toolchains that integrate with [[10 - Cloud, Infra y Backend]] services.

The unifying theme is **conditioning**: moving from unconditional or text-only generation to multi-source guidance where geometry, style, and subject identity are independently controllable. By the end of this note, you will be able to choose the right conditioning strategy for any generative task: ControlNet for structural constraints, LoRA for style consistency, IP-Adapter for reference-based generation, and SDXL/Flux for maximum quality.

---

## 1. ControlNet: Spatial Conditioning

### The Zero-Convolution Trick

ControlNet (Zhang et al., 2023) solves the controllability problem in diffusion models. Standard diffusion accepts only text embeddings as conditioning, which is inherently ambiguous for spatial tasks. ControlNet introduces a trainable copy of the UNet encoder that accepts an additional conditioning image—edges, depth maps, or human poses—and injects its outputs into the frozen base UNet via **zero-initialized convolution layers**.

The zero-convolution is a $1 \times 1$ convolution initialized with weights and bias set to zero:

$$\text{ZeroConv}(x) = W \cdot x + b, \quad W = 0, \ b = 0$$

At the start of training, $\text{ZeroConv}(x) = 0$ regardless of $x$, so the base model behaves exactly as if ControlNet did not exist. During training, gradients flow through ZeroConv and the copied encoder, gradually learning to produce meaningful conditioning signals. This preserves the rich prior of the pre-trained diffusion model while teaching it to respect the spatial structure of the conditioning input.

Formally, the base UNet has locked parameters $\Theta$ and the ControlNet has trainable parameters $\Theta_c$. If the base UNet processes a latent $z$ with conditioning $c$ as $F(z; \Theta, c)$, then with ControlNet:

$$F_{\text{control}}(z, c_{\text{cond}}; \Theta, \Theta_c) = F(z; \Theta, c) + \text{ZeroConv}(\text{Enc}_{\Theta_c}(c_{\text{cond}}))$$

The result is a plug-and-play mechanism. A single base Stable Diffusion model works with different ControlNets for canny edges, depth maps, normal maps, scribbles, or open poses. Each ControlNet is a relatively small checkpoint (~800 MB) compared to the base model (~4 GB), making it practical to hot-swap conditioning types at runtime.

```python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png")
image = cv2.Canny(np.array(image), 100, 200)
image = Image.fromarray(image)

result = pipe(
    "a colorful bird, high quality, detailed",
    negative_prompt="low quality, blurry",
    image=image,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]
```

❌ **Antipattern**: Using a ControlNet checkpoint trained for SD v1.5 with SDXL or Flux. The channel dimensions and attention head configurations differ between model versions, causing shape mismatch errors at the injection points.

✅ **Correct**: Always match the ControlNet version to the base model—SD v1.5 ControlNets work only with SD v1.5, SDXL ControlNets only with SDXL.

💡 **Tip**: Preprocess your conditioning image to match the base model's training resolution (512×512 for SD v1.5, 1024×1024 for SDXL). The pipeline resizes automatically, but preprocessing ensures edge maps don't lose fine details during scaling.

### Conditioning Types

ControlNet supports multiple conditioning modalities, each requiring a different preprocessing step:

| Condition | Preprocessing | Best For | Model ID |
|-----------|--------------|----------|----------|
| Canny edges | `cv2.Canny(image, 100, 200)` | Line art, sketches, architecture | `lllyasviel/sd-controlnet-canny` |
| Depth | MiDaS depth estimation | 3D-consistent scenes, geometry | `lllyasviel/sd-controlnet-depth` |
| OpenPose | OpenPose skeleton extraction | Human poses, character animation | `lllyasviel/sd-controlnet-openpose` |
| Scribble | Human-drawn sketch | Rough ideas, concept art | `lllyasviel/sd-controlnet-scribble` |
| Normal map | Normal map extraction | Surface detail, lighting | `lllyasviel/sd-controlnet-normal` |

![ControlNet Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/ControlNet_diagram.png/800px-ControlNet_diagram.png)

**Caso real**: Interior AI uses ControlNet depth conditioning to let users upload room photos and generate redesigns that respect the existing geometry. The depth map encodes 3D layout—walls, furniture, windows—so the generated redesign maintains the same spatial relationships. Pure text prompts cannot achieve this structural consistency, making ControlNet the key enabling technology for virtual staging.

---

## 2. LoRA, IP-Adapter, and Personalization

### Low-Rank Adaptation for Diffusion

Full fine-tuning of a 4 GB diffusion model for every new style or subject is prohibitively expensive. Low-Rank Adaptation (LoRA, Hu et al., 2021) adapts the idea from NLP to diffusion: instead of updating all UNet weights, LoRA injects trainable low-rank matrices into the cross-attention layers.

If the original weight matrix is $W \in \mathbb{R}^{d \times k}$, LoRA represents the update as:

$$\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, \ A \in \mathbb{R}^{r \times k}$$

where $r \ll \min(d, k)$, typically $r = 4$–$64$. During inference, the adapted forward pass is:

$$h = Wx + \frac{\alpha}{r} \cdot BAx$$

where $\alpha$ scales the LoRA contribution. The rank $r$ controls expressivity: higher ranks capture more detail but increase checkpoint size. A typical style LoRA has $r=16$ and produces a checkpoint of ~10–50 MB—100x smaller than a full fine-tuned model.

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("nerijs/pixel-art-xl")
pipe.fuse_lora()

image = pipe(
    "a castle in the mountains, pixel art style",
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

pipe.unfuse_lora()
```

❌ **Antipattern**: Loading multiple LoRA weights without understanding their interaction. LoRA weights are additive; loading an "anime style" LoRA and a "photorealistic" LoRA simultaneously produces conflicting attention biases, often resulting in washed-out or conflicting outputs.

✅ **Correct**: Load one LoRA at a time using `pipe.unload_lora_weights()` before switching. For combined effects, use LoRA merging scripts that interpolate weights at a ratio you control.

### IP-Adapter: Image Prompting

IP-Adapter (Ye et al., 2023) introduces image prompting via decoupled cross-attention. Instead of noising the entire source image (as in img2img), it extracts image features via a CLIP image encoder and injects them through separate cross-attention layers:

$$h = \text{Attention}(Q, K_{\text{text}}, V_{\text{text}}) + \lambda \cdot \text{Attention}(Q, K_{\text{img}}, V_{\text{img}})$$

where $\lambda$ is the IP-Adapter scale. The decoupled architecture means image and text conditioning operate in parallel—you can combine a text prompt "in the style of Van Gogh" with an image reference for the subject, and the two conditioning signals complement rather than conflict.

```python
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.7)

ip_image = Image.open("reference_portrait.jpg")
image = pipe(
    prompt="a portrait in the style of the reference",
    ip_adapter_image=ip_image,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]
```

❌ **Antipattern**: Forgetting to call `pipe.unload_lora_weights()` before loading a new LoRA or IP-Adapter. The weights accumulate in the UNet's attention layers, causing unpredictable blend of styles.

✅ **Correct**: After using any adapter, unload it explicitly. The mnemonic is **CLEAN HOUSE BETWEEN STYLES**.

### DreamBooth and Textual Inversion

DreamBooth (Ruiz et al., 2022) personalizes diffusion models by fine-tuning on 3–5 images of a specific subject with a unique identifier token (e.g., "a photo of a [V] dog"). Combined with LoRA for parameter efficiency, DreamBooth becomes feasible on a single GPU in under 30 minutes. The prior preservation loss prevents overfitting:

$$\mathcal{L} = \mathbb{E}_{x, c, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right] + \lambda \cdot \mathbb{E}_{x_{\text{prior}}, c_{\text{prior}}, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_{t, \text{prior}}, t, c_{\text{prior}})\|^2 \right]$$

The second term samples from the original model's distribution to prevent the fine-tuned model from forgetting how to generate diverse images. Textual Inversion takes an even lighter approach: it learns only new token embeddings in the text encoder (≈ 10 KB) while freezing the entire UNet, ideal for teaching new concepts without altering generation behavior.

![LoRA Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Low-rank_approximation.png/640px-Low-rank_approximation.png)

**Caso real**: PhotoRoom uses DreamBooth-style fine-tuning to let e-commerce sellers generate product photos with their specific items in various contexts. A seller uploads 5 photos of a handbag, and the model learns to render that exact handbag in different backgrounds, lighting conditions, and angles. Prior to DreamBooth, sellers needed a full photoshoot for each context ($500+); now they generate hundreds of variations in minutes.

---

## 3. SDXL, Flux, and Community Pipelines

### Scaling Up: SDXL

Stable Diffusion XL (Podell et al., 2023) addresses the main limitations of SD v1.5: low resolution (512 px), weak text rendering, and poor composition with complex prompts. SDXL uses:

- **Larger UNet**: 2.6B parameters vs SD v1.5's 860M
- **Dual text encoders**: OpenCLIP ViT-bigG (1.2B) alongside CLIP ViT-L (memory intensive but preserves detailed prompt semantics)
- **Two-stage pipeline**: A base model generates latents at 1024×1024, and a refiner model adds high-frequency details

The dual encoders provide richer semantic understanding. The base model handles composition and subject placement; the refiner focuses on texture, lighting, and fine detail. The refiner operates as an img2img pass with low strength (~0.3):

```python
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16"
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16, variant="fp16"
).to("cuda")

prompt = "a majestic lion wearing a crown, digital art, highly detailed"

latent = pipe(prompt=prompt, num_inference_steps=40, guidance_scale=7.5, output_type="latent").images[0]
image = refiner(prompt=prompt, image=latent, num_inference_steps=20).images[0]
```

### Flow Matching: Flux

Flux (Black Forest Labs, 2024) replaces the UNet with a transformer-based diffusion backbone and uses **flow matching**—a continuous-time formulation of diffusion. Instead of predicting discrete noise at step $t$, flow matching trains the model to predict the velocity $v$ along a straight-line probability path from noise to data:

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t \sim [0,1],\ x_0,\ x_1} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right]$$

where $x_t = (1 - t) \cdot x_0 + t \cdot x_1$ is the linear interpolation between noise $x_0$ and data $x_1$. The model learns to predict the direction vector $x_1 - x_0$ at any point along the path. This formulation is more stable and efficient than discrete-step diffusion, especially at high resolutions.

Flux scales to 12B parameters and natively supports resolutions up to 2 MP without a separate upscaler or refiner. It uses a transformer backbone with rotary positional embeddings (RoPE) and dual-modality attention, processing both text and image tokens in a single sequence.

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="a futuristic cityscape at dusk, neon lights reflecting on wet streets",
    guidance_scale=3.5,
    num_inference_steps=50,
    height=1024,
    width=1024
).images[0]
```

❌ **Antipattern**: Using SDXL with a v1.5 VAE or scheduler. SDXL has its own VAE with a different scaling factor (0.13025 vs 0.18215). Using the wrong VAE produces color-shifted or artifact-ridden images.

✅ **Correct**: Always use the VAE and scheduler bundled with the SDXL checkpoint. The variant files (`vae/diffusion_pytorch_model.safetensors`) are specific to SDXL.

❌ **Antipattern**: Running Flux on GPUs with less than 24 GB VRAM without CPU offloading. Flux's 12B parameter transformer at bfloat16 requires ~24 GB for weights alone, plus activation memory for 1024×1024 inference.

✅ **Correct**: Use `device_map="auto"` or enable model CPU offloading via `pipe.enable_model_cpu_offload()` for Flux on smaller GPUs.

![SDXL Comparison](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Stable_Diffusion_XL_comparison.png/800px-Stable_Diffusion_XL_comparison.png)

### Community Pipelines

Community pipelines in `diffusers` are user-contributed implementations of research papers not yet in the core library. They provide rapid access to cutting-edge techniques:

| Pipeline | Capability | Use Case |
|----------|-----------|----------|
| AnimateDiff | Motion modules for video | Generate short animations from text |
| Latent Consistency Model (LCM) | 4-step inference | Ultra-fast generation for real-time apps |
| UniPC | Unified predictor-corrector | Faster convergence, fewer steps |
| SDEdit | Stochastic editing | Guided image edits with noise injection |

💡 **Tip**: When evaluating SDXL vs Flux for your use case, consider the memory budget. SDXL requires ~12 GB VRAM for 1024×1024 generation; Flux requires ~24 GB for the same resolution. If your production hardware is limited to 16 GB (RTX 4060 Ti / A10), choose SDXL with CPU offloading for the refiner.

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("latent-consistency/lcm-lora-sdv1-5")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()

image = pipe("a cat wearing a hat", num_inference_steps=4, guidance_scale=1.0).images[0]
```

| Architecture | Parameters | Base Resolution | Text Encoders | Backbone |
|-------------|-----------|----------------|---------------|----------|
| SD v1.5 | 860M | 512×512 | CLIP ViT-L (1×) | UNet |
| SDXL | 2.6B | 1024×1024 | CLIP ViT-L + OpenCLIP ViT-bigG (2×) | UNet |
| Flux | 12B | 1024×1024 up to 2MP | T5-XXL + CLIP | Transformer (DiT) |

💡 **Tip**: Community pipelines are loaded via `pipe = DiffusionPipeline.from_pretrained("user/repo-name")`. If a pipeline requires special dependencies, the `diffusers` library detects and prompts you to install them. Always test community pipelines in an isolated environment first—they are user-contributed and may have varying quality and safety guarantees.

**Caso real**: Leonardo.ai uses SDXL as its default generation backbone because the dual text encoders and larger UNet produce significantly better prompt adherence for complex scenes with multiple characters and objects. They benchmarked SDXL against SD v1.5 on 1000 complex prompts and found a 40% reduction in user rating failures (images that did not match the prompt) while maintaining a 3-second inference budget on A100 GPUs. The dual encoder strategy was particularly effective for prompts involving spatial relationships ("a cat on the left and a dog on the right") which SD v1.5 frequently mis-rendered.

---

## 📦 Compression Code

```python
"""Advanced diffusers: ControlNet, LoRA, IP-Adapter, SDXL, Flux."""
import torch
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel,
    StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
    DiffusionPipeline,
)
from PIL import Image
import cv2
import numpy as np

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")
img = np.array(Image.open("input.png"))
canny = Image.fromarray(cv2.Canny(img, 100, 200))
out_cn = pipe_cn("a beautiful landscape", image=canny, num_inference_steps=20).images[0]
out_cn.save("controlnet_out.png")

pipe_lora = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipe_lora.load_lora_weights("nerijs/pixel-art-xl")
pipe_lora.fuse_lora()
out_lora = pipe_lora("a dragon", num_inference_steps=25).images[0]
out_lora.save("lora_out.png")

base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
latent = base("a futuristic car", num_inference_steps=40, output_type="latent").images[0]
out_sdxl = refiner("a futuristic car", image=latent, num_inference_steps=20).images[0]
out_sdxl.save("sdxl_out.png")
```

## 🎯 Key Takeaways

- ControlNet duplicates the UNet encoder and injects conditioning via zero-convolutions, preserving the frozen base model while adding spatial control.
- Zero-convolutions start at zero output, ensuring ControlNet does not disrupt the pre-trained model at initialization.
- LoRA enables efficient style/subject adaptation with ~10–50 MB checkpoints instead of 4 GB full fine-tuned models.
- IP-Adapter uses decoupled image cross-attention for image prompting without altering the latent noise initialization.
- DreamBooth personalizes models with 3–5 images using prior preservation loss to prevent catastrophic forgetting.
- SDXL uses dual text encoders and a base + refiner two-stage pipeline for 1024×1024 generation with superior prompt adherence.
- Flux replaces the UNet with a transformer backbone and flow matching for continuous-time generation at up to 2 MP.
- Community pipelines in `diffusers` provide rapid access to cutting-edge research (AnimateDiff, LCM, UniPC).

## References

- Zhang et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV.
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
- Ruiz et al. (2022). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR.
- Gal et al. (2022). "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." ICLR.
- Ye et al. (2023). "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models." arXiv.
- Podell et al. (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." ICLR.
- Esser et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." (Flux).
- HuggingFace Diffusers Documentation: https://huggingface.co/docs/diffusers
