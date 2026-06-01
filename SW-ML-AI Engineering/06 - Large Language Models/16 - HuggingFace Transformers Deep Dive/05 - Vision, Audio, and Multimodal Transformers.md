# 🤗 Vision, Audio, and Multimodal Transformers

## 🎯 Learning Objectives

- Master the `AutoImageProcessor` and `AutoProcessor` APIs for vision and audio inputs
- Understand how Vision Transformers (ViT, DeiT, DETR) work inside the `transformers` library
- Use `pipeline()` for vision tasks: `image-classification`, `object-detection`, and `automatic-speech-recognition`
- Fine-tune and deploy Whisper and Wav2Vec2 for audio understanding and generation
- Integrate CLIP, BLIP, and LLaVA via `transformers` classes for multimodal reasoning
- Bridge theoretical concepts from [[05 - Deep Learning y CV]] with practical HuggingFace tooling
- Design end-to-end pipelines that combine vision, audio, and text in a single inference graph

---

## Introduction

The transformer architecture, originally designed for natural language processing, has become the universal substrate of modern deep learning. While [[06 - Large Language Models]] focuses on text-centric transformers, the real power of the HuggingFace ecosystem lies in generalizing the same abstractions—`AutoModel`, `AutoTokenizer`, `pipeline`—to images, audio, and multimodal data. This note extends the practitioner's toolkit beyond `AutoModelForCausalLM` into the landscape of `AutoImageProcessor`, `AutoProcessor`, and `AutoModelForObjectDetection`.

This note is essential for ML engineers building perception systems: computer vision pipelines for autonomous vehicles, speech recognition for voice assistants, and multimodal agents that reason over visual scenes and natural language instructions. The `transformers` library unifies these under a single API, but each modality carries unique preprocessing semantics, tensor shapes, and post-processing requirements. Understanding these distinctions prevents the common mistake of treating a Mel-spectrogram like a token ID sequence.

The content here directly complements [[09 - MLOps y Produccion]] and [[10 - Cloud, Infra y Backend]] when deploying multimodal endpoints at scale. Each modality has its own data loading patterns, batching constraints, and hardware affinities—vision benefits from GPU tensor cores, audio from CPU-based resampling, and multimodal from careful I/O orchestration.

The three core abstractions you will use are `AutoImageProcessor` (vision preprocessing), `AutoProcessor` (multimodal preprocessing that may wrap a tokenizer + image processor), and `pipeline()` (unified inference across all tasks). The choice between them depends on whether you need fine-grained control or rapid prototyping:

| Abstraction | Control Level | Use Case |
|-------------|--------------|----------|
| `pipeline()` | Low—automatic preprocessing | Prototyping, demos, simple prod |
| `AutoImageProcessor` + `AutoModel*` | High—explicit tensor shapes | Production with custom batching |
| `Trainer` | Full training loop control | Fine-tuning on custom datasets |

---



## 1. Vision Transformers

### Patches as Tokens

Before Vision Transformers (ViT), computer vision was dominated by CNNs, which encode inductive biases of locality and translation invariance through sliding kernels. While effective, this constrains the receptive field and makes long-range spatial reasoning expensive. A CNN requires many layers to relate pixels in opposite corners of an image; a transformer does it in a single self-attention layer.

The ViT paper (Dosovitskiy et al., 2020) asked a radical question: what if we treat an image as a sequence of patches and feed them into a standard transformer encoder? Given an image $I \in \mathbb{R}^{H \times W \times C}$, we split it into $N = HW / P^2$ patches of size $P \times P$, flatten each, and project linearly to dimension $D$:

$$x_p = W_e \cdot \text{Flatten}(\text{Patch}_p(I)) \quad \text{for } p = 1, \ldots, N$$

where $W_e \in \mathbb{R}^{D \times (P^2 C)}$ is a learnable projection matrix. A learnable `[CLS]` token is prepended, and positional embeddings are added to retain spatial information. The sequence $[z_0; z_1; \ldots; z_N]$ passes through $L$ transformer encoder layers using multi-head self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D_k}}\right)V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$. The `[CLS]` token's final hidden state serves as the image representation for classification, passed through a small MLP head. At sufficient scale, ViT outperformed state-of-the-art CNNs on ImageNet, proving that pure attention can replace convolution.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class_idx = outputs.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

❌ **Antipattern**: Manually resizing the image or normalizing with ad-hoc mean/std values. The input distribution shifts silently, producing wrong logits with high confidence.

✅ **Correct**: Use `AutoImageProcessor` which applies the exact preprocessing pipeline—resize to `processor.size`, center crop, normalize with the checkpoint's mean and std—that the model was trained with.

💡 **Tip**: Inspect `processor.size` and `model.config.image_size` before inference. ViT has never seen pixels outside its training canvas. For example, `google/vit-base-patch16-224` expects exactly $224 \times 224$ pixels.

You can also use the `pipeline` API for a more concise interface:

```python
from transformers import pipeline

pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
result = pipe(Image.open("cat.jpg"))
print(result)
```

This handles loading, preprocessing, inference, and softmax all in one call. The downside: you lose control over preprocessing details, which matters for debugging edge cases. For production, prefer the explicit `AutoImageProcessor` + `AutoModel*` pattern so you control batching and tensor shapes.

### Fine-Tuning Vision Models

For custom datasets, the `Trainer` API works identically for vision models. The main difference is the data collator: instead of `DataCollatorWithPadding`, you pass raw pixel values directly. Use `AutoModelForImageClassification` with `num_labels` set to your dataset's class count:

```python
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=5,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./vit-finetuned",
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    remove_unused_columns=False
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

### DETR: Object Detection as Set Prediction

DETR (Carion et al., 2020) reframes object detection as a set prediction problem, eliminating hand-crafted components like anchor boxes and non-maximum suppression. A CNN backbone extracts features, a transformer encoder processes them, and a decoder uses 100 learned object queries to predict bounding boxes and class labels in parallel. The Hungarian loss matches predictions to ground truth as a bipartite matching problem, making the entire pipeline end-to-end differentiable.

The loss combines class probabilities and bounding box regression:

$$\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \left[-\log \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \emptyset} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})\right]$$

where $\sigma$ is a permutation of predictions, $\hat{p}_{\sigma(i)}(c_i)$ is the predicted probability for ground truth class $c_i$, and $\mathcal{L}_{\text{box}}$ combines L1 and GIoU losses.

```python
from transformers import AutoModelForObjectDetection, pipeline
import torch

detr = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
results = pipe(Image.open("street_scene.jpg"))

for r in results:
    print(f"{r['label']}: score={r['score']:.2f}, box={r['box']}")
```

```python
# Manual control for production pipelines
detr = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
pixel_values = processor(images=Image.open("street_scene.jpg"), return_tensors="pt").pixel_values

with torch.no_grad():
    outputs = detr(pixel_values)

logits = outputs.logits
probs = logits.softmax(-1)[0, :, :-1]
scores, labels = probs.max(-1)

for i in range(len(scores)):
    if scores[i] > 0.7:
        print(f"Label: {detr.config.id2label[labels[i].item()]}, Score: {scores[i]:.2f}")
```

![Vision Transformer Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Vision_Transformer_%28ViT%29_Architecture.png/800px-Vision_Transformer_%28ViT%29_Architecture.png)

DeiT (Touvron et al., 2021) showed that data-efficient training of ViTs is possible with strong regularization and a CNN teacher through knowledge distillation. The student ViT minimizes a combination of ground truth cross-entropy and KL divergence against the teacher's soft predictions. With these techniques, DeiT achieves competitive accuracy using only ImageNet-1k (1.3M images) rather than the massive JFT-300M dataset used by the original ViT.

❌ **Antipattern**: Using a regular ViT for dense prediction tasks like segmentation. ViT's patch-level outputs are coarse; DETR's transformer decoder or specialized backbones (like Swin) are better suited.

✅ **Correct**: Match the architecture to the task. Use `AutoModelForImageClassification` for classification, `AutoModelForObjectDetection` for detection, and task-specific models for segmentation.

**Caso real**: Tesla's vision-only autopilot uses transformer-based detection backbones to reason globally over camera feeds. A car approaching from a distant lane occupies only a few pixels, but self-attention connects it to the full scene context without the limited receptive field of convolutions. This eliminates the need for hand-tuned anchor configurations and multi-scale feature pyramids that plague CNN-based detectors.

---

## 2. Audio Transformers

### Self-Supervised Speech with Wav2Vec2

Audio signals are one-dimensional time-series with complex temporal structure spanning milliseconds to minutes. Early deep learning for audio relied on hand-crafted spectrograms fed into CNNs or RNNs. Wav2Vec2 (Baevski et al., 2020) introduced self-supervised pretraining for speech: a convolutional feature encoder processes raw waveform into latent representations, then a transformer encoder operates on the masked feature sequence with a contrastive objective.

Wav2Vec2's training objective mirrors BERT's masked language modeling but in the continuous acoustic domain. The model is trained to distinguish true quantized speech features from distractor negatives at masked positions:

$$\mathcal{L} = -\sum_{t \in \mathcal{M}} \log \frac{\exp(\text{sim}(c_t, q_t)/\tau)}{\sum_{\tilde{q} \sim Q} \exp(\text{sim}(c_t, \tilde{q})/\tau)}$$

where $c_t$ is the context representation from the transformer, $q_t$ is the quantized target feature, $\mathcal{M}$ is the set of masked timesteps, and $Q$ is a set of distractors sampled from the same utterance.

```python
from transformers import AutoModelForAudioClassification
from datasets import Audio, load_dataset
import torch

model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
processor = AutoImageProcessor.from_pretrained("facebook/wav2vec2-base")

dataset = load_dataset("superb", "ic", split="test")
sample = dataset[0]["audio"]

inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(-1).item()
```

The key advantage: Wav2Vec2 can be fine-tuned on very small labeled datasets (hours instead of thousands of hours) because it has already learned robust speech representations through self-supervision.

### Whisper: Seq2Seq Transcription

Whisper (Radford et al., 2022) takes a different approach. Instead of self-supervised learning, it uses massive weakly supervised training on 680,000 hours of multilingual, multitask audio. It frames ASR as a straightforward sequence-to-sequence problem: an encoder processes log-Mel spectrograms (80 filterbanks over 25ms windows with 10ms stride), and a decoder generates text tokens autoregressively.

The log-Mel spectrogram is a time-frequency representation where the frequency axis is warped to the Mel scale, which approximates human auditory perception:

$$M_{t,f} = \log\left(\sum_{k} |X_{t,k}|^2 \cdot H_{f,k}\right)$$

where $X_{t,k}$ is the STFT magnitude at time $t$ and frequency bin $k$, and $H_{f,k}$ warps to Mel scale via triangular filterbanks.

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")

audio_array, sampling_rate = librosa.load("audio.mp3", sr=16000)
inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
predicted_ids = model.generate(inputs["input_features"], forced_decoder_ids=forced_decoder_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
```

❌ **Antipattern**: Feeding 44.1 kHz audio to Whisper without resampling to 16 kHz. The Mel-spectrogram frequency bins are designed for the 0–8 kHz range (at 16 kHz sampling). Upsampled audio shifts these bins, producing frequency-mapped gibberish.

✅ **Correct**: Always resample to 16 kHz using `librosa.load(path, sr=16000)` or load with `datasets.Audio(sampling_rate=16000)`.

💡 **Tip**: The `pipeline` API for ASR handles resampling, chunking, and decoding automatically:

```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
result = pipe("long_audio.mp3", chunk_length_s=30, return_timestamps=True)
```

Use `chunk_length_s` for files longer than the model's context window. Whisper processes ~30-second chunks; overlapping chunks prevent word truncation.

![Spectrogram](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Spectrogram-19thC.png/640px-Spectrogram-19thC.png)

❌ **Antipattern**: Using `AutoModelForAudioClassification` for transcription tasks. The classification head outputs discrete labels (emotions, commands), not text sequences. For ASR, use `AutoModelForSpeechSeq2Seq` with an autoregressive decoder.

✅ **Correct**: Check the model card. If the task is `automatic-speech-recognition`, use `AutoModelForSpeechSeq2Seq` or the `pipeline` abstraction. If it's `audio-classification`, use `AutoModelForAudioClassification`.

**Caso real**: Descript uses Whisper-like models to transcribe and edit audio by editing text. Users can delete a spoken phrase from the transcript and the underlying waveform is automatically modified. This non-linear audio editing paradigm reduces hours of waveform editing to minutes of text editing. The feature works by aligning transcription timestamps with waveform regions and applying cross-fade operations on the audio backend.

### SpeechT5 and Unified Speech Processing

SpeechT5 unifies text-to-speech and speech-to-text in a shared encoder-decoder framework. The encoder processes speech or text inputs into a common hidden representation, and task-specific decoders (a linear layer for speech, a spectrogram decoder for TTS, a text decoder for ASR) generate the appropriate output modality. The pre-training objective combines speech reconstruction and text denoising, forcing the encoder to learn modality-agnostic representations.

The HuggingFace `transformers` library exposes SpeechT5 through `SpeechT5ForTextToSpeech` and `SpeechT5ForSpeechToText`. The `AutoProcessor` generates both text token IDs and speech features (log-Mel spectrograms) from the same input:

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

inputs = processor(text="Hello, welcome to speech synthesis", return_tensors="pt")
# Generate speech from text using a speaker embedding
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_emb)
```

Each audio model type uses a different processor. Know which one you need:

| Model | `AutoModelFor*` Class | Input |
|-------|----------------------|-------|
| Wav2Vec2 | `AudioClassification` | Raw waveform → latent features |
| Whisper | `SpeechSeq2Seq` | Raw waveform → log-Mel → text |
| SpeechT5 TTS | `SpeechT5ForTextToSpeech` | Text → spectrogram → waveform |

---

## 3. Multimodal Reasoning

### CLIP and Contrastive Alignment

Multimodal learning addresses a fundamental limitation of unimodal models: the world is not composed of text alone. CLIP (Radford et al., 2021) demonstrated that joint training on 400 million image-text pairs creates a shared embedding space where semantic similarity is preserved across modalities. The training uses a contrastive objective in a batch of $N$ image-text pairs:

$$\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^N \left[\log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j)/\tau)} + \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_j, T_i)/\tau)}\right]$$

where $\text{sim}(I, T) = E_v(I) \cdot E_t(T) / (\|E_v(I)\| \|E_t(T)\|)$ is the cosine similarity. The symmetric loss pushes matched pairs together along both image→text and text→image directions. The temperature $\tau$ controls the sharpness of the softmax distribution.

This enables zero-shot classification: compute the image embedding and compare it against text embeddings of candidate labels:

$$p(y=k|x) = \frac{\exp(\text{sim}(E_v(x), E_t(\text{prompt}_k)) / \tau)}{\sum_{j} \exp(\text{sim}(E_v(x), E_t(\text{prompt}_j)) / \tau)}$$

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

❌ **Antipattern**: Using CLIP for fine-grained classification like "Siberian Husky" vs "Alaskan Malamute" without descriptive prompts. The embedding shifts subtly between similar breeds; without context, CLIP defaults to the most frequent visual pattern.

✅ **Correct**: Use prompt engineering. Instead of `"Siberian Husky"`, use `"a photo of a Siberian Husky, a type of dog"`. The additional context anchors the embedding in the visual domain.

💡 **Tip**: CLIP's `processor` handles both image preprocessing and text tokenization in one call. Never use `AutoImageProcessor` for CLIP—you need `CLIPProcessor` which wraps both modalities.

### BLIP, LLaVA, and Visual Instruction Following

CLIP excels at ranking but cannot generate text. BLIP (Li et al., 2022) unified vision-language understanding and generation through a multimodal mixture of encoder-decoder objectives: captioning (image→text), filtering (text→image relevance), and understanding (VQA). It introduces a captioner and a filter trained jointly so the filter learns to remove noisy web text while the captioner generates high-quality descriptions.

LLaVA (Liu et al., 2023) pushed further by connecting a pre-trained CLIP vision encoder to a large language model (Vicuna) via a simple linear projector. The projector maps visual tokens into the LLM's embedding space, enabling the language model to reason about images as if they were part of the text sequence:

$$h_{\text{vision}} = W \cdot E_v(I) \quad \text{(projection layer)}$$
$$p(y | h_{\text{vision}}, h_{\text{text}}) = \text{LLM}_{\theta}([h_{\text{vision}}; h_{\text{text}}])$$

The key insight: the projector $W \in \mathbb{R}^{D_{\text{llm}} \times D_{\text{clip}}}$ is the only trained component connecting modalities; the vision encoder and LLM remain frozen. This makes LLaVA extremely parameter-efficient to train.

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"
inputs = processor(text=prompt, images=Image.open("scene.jpg"), return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

![CLIP Concept](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Contrastive_Language-Image_Pre-training.png/800px-Contrastive_Language-Image_Pre-training.png)

❌ **Antipattern**: Using `AutoImageProcessor` or `AutoTokenizer` separately for multimodal models. The `AutoProcessor` for multimodal models wraps both, ensuring consistent image dimensions and tokenization alignment.

✅ **Correct**: Use `AutoProcessor.from_pretrained("model-id")` for any multimodal model. It returns the correct processor class (e.g., `CLIPProcessor`, `LlavaProcessor`) with all sub-processors configured.

💡 **Tip**: For production deployments, pre-compute CLIP image embeddings and store them in a vector database (FAISS, Pinecone). At query time, only the text embedding needs to be computed, reducing inference latency from O(N) to O(log N) via approximate nearest neighbor search.

**Caso real**: E-commerce platforms like eBay use CLIP-based image-text retrieval to enable "search by image" and "search by natural language description" in the same interface. A user uploads a photo of a mid-century chair and the system returns visually similar listings, or types "green velvet sofa under $500" to retrieve matching products. The shared embedding space handles both modalities without separate models for visual similarity and text search.

### Idefics2 and Unified Multimodal Models

Idefics2 (HuggingFace, 2024) represents the next generation of multimodal models that process interleaved images and text in a single sequence. Unlike CLIP (embedding space only) or LLaVA (single image + single text prompt), Idefics2 can handle multiple images in arbitrary positions within the text. It builds on Mistral's architecture with a perceiver resampler that compresses visual features before injecting them into the language model:

```python
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
from PIL import Image

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image in detail."}
    ]},
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[Image.open("photo.jpg")], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
```

The perceiver resampler is key: it downsamples $N$ visual tokens (e.g., 729 from a ViT) to $M$ compact tokens (e.g., 64) via cross-attention, making it computationally feasible to fit multiple images into the LLM's context window. This architecture now dominates production multimodal systems.

---

## 📦 Compression Code

```python
"""Unified multimodal inference: vision, audio, and CLIP."""
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import librosa

vision_pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
result = vision_pipe(Image.open("photo.jpg"))
print("Vision:", result[0])

audio, sr = librosa.load("speech.mp3", sr=16000)
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
print("Audio:", asr_pipe(audio))

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = clip_proc(text=["a cat", "a dog"], images=Image.open("photo.jpg"), return_tensors="pt", padding=True)
probs = clip_model(**inputs).logits_per_image.softmax(dim=1)
print("Multimodal:", probs)
```

## 🎯 Key Takeaways

- `AutoImageProcessor` abstracts all vision preprocessing (resize, normalize, tensorize) using model-specific statistics from the checkpoint.
- ViT treats images as patch sequences with self-attention replacing convolution; DETR removes anchor-NMS complexity via transformer set prediction.
- Whisper frames ASR as seq2seq translation on log-Mel spectrograms; always resample audio to 16 kHz and use `forced_decoder_ids`.
- Wav2Vec2 uses self-supervised contrastive learning on masked speech representations, analogous to BERT's MLM objective.
- CLIP creates a shared image-text embedding space via symmetric contrastive learning, enabling zero-shot classification.
- Multimodal models (BLIP, LLaVA) connect vision encoders to language decoders via lightweight projector layers.
- The `pipeline()` API provides one-line inference but hides preprocessing details—understand them before production use.
- Match the `AutoModel*` class to the task (classification, detection, seq2seq, multimodal) to get the correct architecture head.

## References

- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.
- Carion et al. (2020). "End-to-End Object Detection with Transformers." ECCV.
- Touvron et al. (2021). "Training data-efficient image transformers & distillation through attention." ICML.
- Baevski et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS.
- Radford et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." ICML.
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
- Li et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training." ICML.
- Liu et al. (2023). "Visual Instruction Tuning." NeurIPS.
- HuggingFace Transformers Documentation: https://huggingface.co/docs/transformers
