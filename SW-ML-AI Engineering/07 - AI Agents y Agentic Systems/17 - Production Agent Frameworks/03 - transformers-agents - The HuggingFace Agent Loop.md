# 🤗 transformers.agents — The HuggingFace Agent Loop

## 🎯 Learning Objectives

- Understand the **HuggingFace-native agent loop** built into the `transformers` library, and how it differs from smolagents (same team, different API)
- Use **`HfApiEngine`** to drive agents with any chat model on the HuggingFace Hub, including open-source models
- Build **custom tools with the `@tool` decorator** and the `Tool` base class, with built-in support for **multi-modal tools** (image, audio, video)
- Run the **`react.json` agent** for code-as-action workflows, or the **`react.yaml` / `react.md` agents** for chat and document-grounded workflows
- Wire `transformers.agents` into **Gradio for a local UI** in 5 lines, useful for portfolio demos and customer-facing prototypes
- Combine with [[../16 - OpenShell and Agent Sandboxes/00 - Welcome to OpenShell and Agent Sandboxes.md|OpenShell]] for production sandboxing of agent-generated code

---

## Introduction

`transformers.agents` is the agent framework that ships inside the `transformers` library — the same one you use to load any HuggingFace model, run inference, fine-tune, and export. Released in 2024 and continuously extended through 2026, it is the framework that gives you the **HuggingFace ecosystem as a first-class agent backend**: any of the 1M+ models on the Hub can be wrapped as a tool, and any Hub chat model can be the agent's reasoning engine. The `transformers` library is the standard Python interface to the open-source ML world; `transformers.agents` makes it the standard interface to the open-source agent world.

The framework is built by the same team that built `smolagents` (note 01) and the two share the same `@tool` decorator, the same code-as-action philosophy, and the same `react` prompt format. The difference is the **integration surface**: `transformers.agents` is part of the 200k+ star `transformers` library, so the tools can be Hub models (`image-generation`, `document-question-answering`, `text-to-speech`, `image-captioning`) and the UI is Gradio. `smolagents` is a standalone library that is easier to install and lighter to import, but the tool ecosystem is whatever you write in Python. For your portfolio, the rule is: **use `transformers.agents` when the agent needs a HuggingFace Hub model as a tool; use `smolagents` when the agent is pure-Python**.

The two frameworks are not competitors. The smolagents team explicitly designed the two to be drop-in compatible: the same `@tool` definition works in both, and the same `react.json` prompt is consumed by both. The decision is which surface you want — the standalone `smolagents` library or the integrated `transformers.agents` library. For the **Multi-Agent Research System** portfolio project, the choice is `transformers.agents` if you want the research node to call a `text-to-image` model on the Hub to generate diagrams; otherwise, `smolagents` is the lighter pick.

---

## 1. The Problem and Why This Solution Exists

### 1.1 The Hub-as-toolbox problem

The HuggingFace Hub has 1M+ models, but until 2024 there was no standard way to use them as agent tools. If you wanted an agent that could "look at this image and answer a question about it," you had to write the model loading code, the inference code, the pre/post-processing, and the tool schema — and you had to do it again for every new model you wanted to add. `transformers.agents` solves this with a **tool catalog** that maps task names to Hub models: `image-captioning`, `document-question-answering`, `image-transformation`, `text-to-speech`, `text-to-image`, `text-to-video`. The framework handles the loading, the inference, the cleanup, and the tool schema.

```python
# WITHOUT transformers.agents: 30+ lines of glue code per Hub model
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image_path: str) -> str:
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# WITH transformers.agents: 1 line
from transformers.agents import load_tool
captioner = load_tool("image-captioning", model="Salesforce/blip-image-captioning-base")
caption = captioner(image_path="/tmp/photo.jpg")
```

The 1-line version is the value proposition: the framework knows how to load, run, and clean up the model, and the tool schema is generated automatically. For your **StayBot** Airbnb agent, this means you can add "show me what this apartment looks like" with one line: `image_description = image_captioner(photo_path)`.

### 1.2 The open-model-as-reasoning-engine problem

The other half of the equation is the **reasoning engine** — the LLM that drives the agent loop. Most agent frameworks assume you use GPT-4o or Claude Sonnet, which are the best at tool calling as of 2026. But the best models are also the most expensive, the most rate-limited, and the most locked to a single provider. `transformers.agents` ships **`HfApiEngine`**, a thin wrapper around the HuggingFace Inference API that works with any chat model on the Hub — including open-source models like `meta-llama/Llama-3.3-70B-Instruct`, `mistralai/Mistral-Large-2`, and `Qwen/Qwen3-235B`. The same agent code that runs on GPT-4o runs on an open-source model with one parameter change.

```python
# Provider-agnostic: swap GPT-4o for an open-source Llama 3.3 with one change
from transformers.agents import HfApiEngine, ReactCodeAgent

engine = HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")
agent = ReactCodeAgent(tools=[search_web, calculator], llm_engine=engine)
result = agent.run("What is 17 * 23?")
```

The trade-off: open-source models are weaker at tool calling than GPT-4o or Claude Sonnet. A complex 10-step agent loop may fail on Llama 3.3 70B where it would succeed on Claude Sonnet 4.5. The framework supports `model="claude-3-5-sonnet"` (via the Hub's API proxy) and `model="gpt-4o"` (also via the Hub's API proxy) for the cases where you want the best tool-calling capability without giving up the Hub-native tool catalog.

### 1.3 The code-as-action philosophy

Like smolagents, `transformers.agents` supports **code-as-action** via the `ReactCodeAgent` class. The agent writes Python code, the framework executes it (in a `LocalPythonExecutor` or a custom executor), and the result is fed back. The code-execution surface is the same security concern as smolagents, and the same [[../16 - OpenShell and Agent Sandboxes/00 - Welcome to OpenShell and Agent Sandboxes.md|OpenShell]] or E2B sandbox is the production answer. The framework supports a custom executor via the `executor` parameter on `ReactCodeAgent.__init__`.

---

## 2. Conceptual Deep Dive

### 2.1 The agent classes

`transformers.agents` ships three agent classes, each optimized for a different prompt format and use case:

| Class | Prompt format | Best for | Output |
|-------|---------------|----------|--------|
| `ReactCodeAgent` | `react.json` (code blocks) | Composable multi-step workflows | Python code execution |
| `ReactJsonAgent` | `react.json` (JSON tool calls) | Fast sequential tool calls | Tool return values |
| `ReactTextAgent` | `react.md` (markdown actions) | Document-grounded chat, free-form | Text chunks |

The two `react` agents are similar in spirit to smolagents' `CodeAgent` and `ToolCallingAgent`. The `text` agent is a different beast: it does not execute code or call functions, it produces structured text with citations and is designed for retrieval-augmented chat.

### 2.2 The `@tool` decorator

The decorator is the same as smolagents' (note 01), with one addition: the `inputs` and `outputs` types can include `image`, `audio`, and `video` for multi-modal tools.

```python
from transformers.agents import tool
from PIL import Image

@tool
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of a text as positive, negative, or neutral.

    Args:
        text: The text to classify
    """
    # ... implementation
    return "positive"

@tool
def describe_image(image: Image.Image) -> str:
    """Generate a caption for an image.

    Args:
        image: A PIL Image to caption
    """
    # The agent can pass a PIL Image directly; the framework handles the conversion.
    return "A photo of a cat sitting on a windowsill"
```

The multi-modal support is the differentiator: the tool signature can include `PIL.Image.Image` and the framework will pass the image as a properly formatted input to the chat model. For the **Multi-Agent Research System** portfolio project, the research node can read arXiv papers and pass the figures to a `describe_image` tool, generating inline figure captions without manual pre-processing.

### 2.3 Built-in tool catalog

The framework ships a catalog of ready-to-use tools, all backed by Hub models:

| Tool | Task type | Default model |
|------|-----------|---------------|
| `image-captioning` | Vision | `Salesforce/blip-image-captioning-base` |
| `document-question-answering` | Document QA | `impira/layoutlm-document-qa` |
| `image-transformation` | Image editing | `stabilityai/stable-diffusion-2-1` |
| `text-to-speech` | Audio | `microsoft/speecht5_tts` |
| `text-to-image` | Generation | `stabilityai/stable-diffusion-xl-base-1.0` |
| `text-to-video` | Video | `ali-vilab/text-to-video-ms-1.7b` |
| `image-segmentation` | Vision | `facebook/detr-resnet-50-panoptic` |
| `speech-to-text` | Audio | `openai/whisper-large-v3` |
| `translation` | NLP | ` Helsinki-NLP/opus-mt-en-es` |
| `summarization` | NLP | `facebook/bart-large-cnn` |
| `text-classification` | NLP | `distilbert-base-uncased-finetuned-sst-2-english` |
| `image-classification` | Vision | `google/vit-base-patch16-224` |

The tool catalog is the production value: a portfolio demo can show "the agent reads a paper, generates a summary, captions the figures, and answers questions about the diagrams" without writing a single line of model code. The Hub handles the model loading, the inference, and the cleanup.

### 2.4 The HfApiEngine

`HfApiEngine` is the default reasoning engine, wrapping the HuggingFace Inference API. It accepts any chat model on the Hub:

```python
from transformers.agents import HfApiEngine

# Open-source model via the Hub
engine = HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")

# Proprietary model via the Hub's API proxy
engine = HfApiEngine(model="claude-3-5-sonnet-latest")

# Local model via the Hub's inference endpoints
engine = HfApiEngine(model="https://abc123.us-east-1.aws.endpoints.huggingface.cloud/my-model")
```

The Inference API is a paid service with usage-based pricing. For local development, `TransformersEngine` loads the model in-process, but requires a GPU with enough VRAM for the chosen model. For production, the `HfApiEngine` is the standard choice because it offloads the model serving to HuggingFace's infrastructure.

For multi-provider routing, the `HfApiEngine` is one option, but the same `LiteLLMModel` pattern from smolagents (note 01) works for `transformers.agents` via the `llm_engine` parameter — pass any object that has a `generate(messages)` method, and the framework calls it.

### 2.5 The `react.json` prompt format

The agent's system prompt follows the `react.json` format, which instructs the model to alternate between `Thought:` (reasoning) and `Code:` (action) blocks. The framework parses the model's output, extracts the code, executes it, and feeds the result back as an `Observation:` block. The loop continues until the model emits a `Final Answer:` block.

The prompt format is the same one that smolagents uses (note 01), which is why the two frameworks are drop-in compatible. The format is well-tuned for code-emitting models (Code Llama, DeepSeek Coder, GPT-4o, Claude Sonnet) and works less well with base chat models (Llama 3 base, Mistral base) that have not been instruction-tuned.

### 2.6 The Gradio UI integration

The framework ships a `gradio_ui` helper that takes an agent and produces a local Gradio chat UI in 5 lines:

```python
from transformers.agents import ReactCodeAgent, HfApiEngine, gradio_ui

agent = ReactCodeAgent(tools=[...], llm_engine=HfApiEngine(model="..."))
gradio_ui(agent).launch(server_name="0.0.0.0", server_port=7860)
```

The UI is a chat interface with code execution traces visible, which is perfect for portfolio demos. The same agent that runs in the UI runs in production — only the entry point changes. For a customer-facing prototype, the Gradio UI can be deployed on HuggingFace Spaces for free.

---

## 3. Production Reality

### 3.1 The Hub dependency

The tool catalog depends on the Hub: the model files are downloaded from `huggingface.co`, the Inference API calls Hub endpoints, and the tool schemas reference Hub model IDs. This means production deployments need:

- **Network access** to `huggingface.co` (or a self-hosted Hub mirror).
- **API keys** for the Inference API (free tier available, paid for production scale).
- **Model acceptance** of the Hub's license terms (some models require accepting a license before download).

For air-gapped deployments, the framework supports a `HF_HUB_OFFLINE=1` mode that loads models from a local cache. The cache must be populated ahead of time, typically by running the tool once on a connected machine and copying the `~/.cache/huggingface/` directory.

### 3.2 Latency profile

`HfApiEngine` calls the Hub's Inference API, which adds ~100-300ms of network overhead on top of the model's own inference time. For GPT-4o-class models, the total per-step latency is 1-3 seconds. For open-source 70B models, the Hub's dedicated endpoints can be faster than the same model running locally without GPU acceleration.

The local `TransformersEngine` eliminates the network overhead but requires GPU VRAM. A 70B model in fp16 needs ~140GB of VRAM, which is not consumer hardware. A 7B model in fp16 needs ~14GB, which is RTX 4090 territory.

### 3.3 Production case — the HuggingFace research digest bot

HuggingFace's official research digest bot (the one that posts daily on `@huggingface`) uses `transformers.agents` with `ReactCodeAgent`, `HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")`, and a custom tool set: `arxiv_search`, `pdf_download`, `pdf_text_extraction`, `summarization` (using a Hub model), `image_captioning` (for paper figures), and `translation` (for non-English papers). The agent runs daily on a HuggingFace Space, generates the digest, and posts it.

The architecture is the production case study for Hub-native agents: the model serving is Hub's responsibility, the tool set is small and Hub-native, and the UI is Gradio. The cap on the architecture is the daily run, not the model size.

### 3.4 Failure modes

| Failure mode | Symptom | Fix |
|--------------|---------|-----|
| Hub model is gated, license not accepted | `RepositoryNotFoundError` or 403 | Accept the license on the Hub page, then re-run |
| Inference API is down | `HfHubHTTPError` 503 | Add retry with backoff; switch to local `TransformersEngine` |
| Tool returns wrong type | LLM re-prompted, eventually succeeds | Improve tool docstring; add `Field(description=...)` |
| Code execution hangs | Step times out | Set `executor_kwargs={"timeout": 60}` on the executor |
| Multi-modal tool receives wrong format | Type error in tool body | Validate inputs at tool entry; convert to `PIL.Image.Image` explicitly |
| Gradio UI exposed publicly without auth | Anyone can run your agent | Add `auth=(user, pass)` to `.launch()` or use a private Space |

### 3.5 Comparison: transformers.agents vs the other five frameworks

| Framework | Hub-native | Best for | Worst for |
|-----------|:----------:|----------|-----------|
| **transformers.agents** | ✅ Yes | Hub models as tools, multi-modal agents | Air-gapped deployments |
| **smolagents** | ⚠️ Manual | Pure-Python agents, code-as-action | Hub model integration |
| **PydanticAI** | ⚠️ Manual | Type-safe production backends | Hub model integration |
| **OpenAI Agents SDK** | ❌ No | OpenAI-only stacks | Hub models, open-source |
| **Google ADK** | ❌ No | GCP deployments | Non-Google, Hub models |
| **CrewAI 1.0** | ⚠️ Manual | Multi-agent role-playing | Hub-native tool catalog |

---

## 4. Code in Practice

### 4.1 Minimal example: text agent with one tool

```python
# 🤗 MINIMAL: transformers.agents with one tool and an open-source LLM
# Install: pip install transformers[agents] huggingface_hub

import os
from transformers.agents import ReactCodeAgent, HfApiEngine, tool

os.environ["HF_TOKEN"] = "hf_..."  # free token from huggingface.co

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Medellín"
    """
    return f"Weather in {city}: 22°C, partly cloudy"  # mock

engine = HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")
agent = ReactCodeAgent(tools=[get_weather], llm_engine=engine)

result = agent.run("What is the weather in Medellín?")
print(result)  # "The weather in Medellín is 22°C, partly cloudy."
```

### 4.2 Multi-modal: read an image and answer a question

```python
# MULTI-MODAL: agent uses a Hub image-captioning tool
from transformers.agents import ReactCodeAgent, HfApiEngine, load_tool
from PIL import Image

captioner = load_tool("image-captioning", model="Salesforce/blip-image-captioning-base")
vqa = load_tool("document-question-answering")  # can also do image QA

engine = HfApiEngine(model="claude-3-5-sonnet-latest")
agent = ReactCodeAgent(tools=[captioner, vqa], llm_engine=engine)

# The agent will:
# 1. Use captioner to describe the image
# 2. Use vqa to answer the specific question
result = agent.run(
    "Look at /tmp/photo.jpg. What is in the image and is there a cat visible?",
    images=[Image.open("/tmp/photo.jpg")],
)
print(result)
```

### 4.3 Document-grounded chat with `ReactTextAgent`

```python
# DOCUMENT QA: text agent for grounded Q&A over a document
from transformers.agents import ReactTextAgent, HfApiEngine, tool

# Pre-loaded document chunks (in production, from a vector store)
document = """
OpenShell is NVIDIA's agent runtime security layer, released in 2025.
It provides four defense layers: Filesystem, Network, Process, and Inference.
The framework is 88.9% Rust, designed for kernel-level enforcement.
"""

@tool
def retrieve_from_doc(query: str) -> str:
    """Search the document for relevant passages.

    Args:
        query: Search query
    """
    # Naive keyword search; replace with a real retriever in production
    return "\n".join(line for line in document.split("\n") if query.lower() in line.lower())

engine = HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")
agent = ReactTextAgent(tools=[retrieve_from_doc], llm_engine=engine)

result = agent.run("What programming language is OpenShell written in?")
print(result)  # "OpenShell is written in Rust, 88.9% of the codebase."
```

### 4.4 Gradio UI in 5 lines

```python
# GRADIO UI: portfolio-ready local chat interface
from transformers.agents import ReactCodeAgent, HfApiEngine, gradio_ui

@tool
def search_papers(query: str) -> str:
    """Search arXiv for papers matching a query.

    Args:
        query: Search query, e.g. "agent frameworks"
    """
    return f"3 papers found about {query}"  # mock

engine = HfApiEngine(model="claude-3-5-sonnet-latest")
agent = ReactCodeAgent(tools=[search_papers], llm_engine=engine)

# One line to launch a Gradio chat UI on http://localhost:7860
gradio_ui(agent).launch(server_name="0.0.0.0", server_port=7860, share=False)
```

### 4.5 Common pitfalls

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| No `HF_TOKEN` environment variable | `RepositoryNotFoundError` on Hub models | Set `HF_TOKEN=hf_...` from huggingface.co/settings/tokens |
| Tool returns wrong type | LLM re-prompted, eventually succeeds | Add explicit type hints and `Field(description=...)` |
| `LocalPythonExecutor` in production | Untrusted code can read your filesystem | Use [[../16 - OpenShell and Agent Sandboxes/00 - Welcome to OpenShell and Agent Sandboxes.md\|OpenShell]] or E2B as the executor |
| Gradio UI without auth | Anyone can run your agent (and your API quota) | Add `auth=("user", "pass")` to `.launch()` |
| Multi-modal tool receives string path | `PIL.Image.open` not called | Validate the type in the tool body; convert to `PIL.Image.Image` |
| `react.json` model emits malformed code | Loop stuck on validation error | Lower `max_iterations`; use a code-tuned model (Code Llama, DeepSeek Coder) |

> 💡 **Tip**: For portfolio demos, the `gradio_ui` helper is the fastest path from a working agent to a customer-facing prototype. The same agent runs in the UI and in production — only the entry point changes.

---

## 📦 Compression Code

```python
# NOTE: 03 - transformers.agents
# Repo: github.com/huggingface/transformers (Apache-2.0, part of transformers v4.39+)
# Same team as smolagents, drop-in compatible @tool and react.json
# Three agent classes: ReactCodeAgent (code-as-action), ReactJsonAgent (JSON tool calls), ReactTextAgent (document chat)
# Tool catalog: 12+ built-in tools backed by Hub models (captioning, QA, TTS, image gen, translation, etc.)
# Reasoning engine: HfApiEngine (Hub Inference API) supports any chat model on the Hub
# Local option: TransformersEngine loads model in-process, requires GPU
# UI: gradio_ui(agent).launch() in 5 lines for portfolio demos
# Custom executor: pass any Executor instance for production sandboxing (OpenShell, E2B)
# Multi-modal: tool inputs can be PIL.Image.Image, the framework handles the conversion

import os
from transformers.agents import ReactCodeAgent, HfApiEngine, tool, gradio_ui

os.environ["HF_TOKEN"] = "hf_..."

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    Args:
        city: City name
    """
    return f"Weather in {city}: 22°C"

engine = HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")
agent = ReactCodeAgent(tools=[get_weather], llm_engine=engine)
result = agent.run("What is the weather in Medellín?")

# Production UI in 5 lines
# gradio_ui(agent).launch(server_name="0.0.0.0", server_port=7860, auth=("admin", "secret"))
```

## 🎯 Key Takeaways

- **The HuggingFace Hub is the agent's toolbox** — 12+ built-in tools (captioning, QA, TTS, image gen) are one `load_tool()` call away
- **`HfApiEngine` works with any chat model on the Hub**, including open-source Llama, Mistral, and Qwen, plus the proprietary models via the Hub's API proxy
- **The `@tool` decorator and `react.json` format are drop-in compatible with smolagents** — same team, same philosophy, same tool definitions
- **`gradio_ui(agent).launch()` is the fastest path from a working agent to a customer-facing prototype** in 5 lines
- **Multi-modal tools with `PIL.Image.Image` inputs** are first-class — the framework handles the conversion to the chat model's expected format

## References

- transformers.agents documentation: https://huggingface.co/docs/transformers/transformers_agents
- transformers library: https://github.com/huggingface/transformers
- Built-in tool catalog: https://huggingface.co/docs/transformers/transformers_agents#tools
- HfApiEngine: https://huggingface.co/docs/transformers/transformers_agents#model-engine
- Gradio integration: https://huggingface.co/docs/transformers/transformers_agents#displayed-agents
- HuggingFace Inference API: https://huggingface.co/docs/api-inference
- HuggingFace Spaces for agent deployment: https://huggingface.co/docs/hub/spaces
- smolagents (sister library, same team): https://github.com/huggingface/smolagents
