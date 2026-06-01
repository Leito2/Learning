# 🤗 Tokenizers and Data Processing

## 🎯 Learning Objectives

- Understand the algorithmic foundations of BPE, WordPiece, Unigram, and Byte-level BPE.
- Master the `tokenizers` library API for fast, parallelized encoding.
- Configure padding, truncation, and batching strategies for variable-length sequences.
- Leverage `datasets` advanced features (`map`, `filter`, streaming) for scalable data pipelines.
- Choose the correct `DataCollator` variant for classification, language modeling, and seq2seq tasks.

## Introduction

Tokenization is the silent bottleneck of every NLP pipeline. A model cannot consume raw text; it needs fixed-vocabulary integer identifiers. The choice of tokenization algorithm directly impacts vocabulary size, out-of-vocabulary rate, sequence length, and even model bias. While early systems used whitespace or character splitting, modern LLMs rely on subword algorithms that balance vocabulary coverage and sequence efficiency.

This note goes beyond `tokenizer.encode()`. We dissect the Rust-backed `tokenizers` library, compare subword algorithms mathematically, and build production data pipelines using the `datasets` library. These skills are prerequisites for efficient training (see [[03 - Trainer, TrainingArguments, and Distributed Training|Trainer note]]) and for understanding why generation behaves differently across model families (see [[04 - Generation, Decoding, and Structured Output|Generation note]]). If you are coming from computer vision, think of tokenization as the analog of image normalization and patch extraction.

---

## 1. Subword Tokenization Algorithms

Subword tokenization emerged because pure word-level vocabularies explode in size (English alone has >1M words) while character-level sequences become too long for transformers to model effectively. The core idea is to represent rare words as concatenations of frequent subword units, keeping the vocabulary compact while maintaining semantic granularity.

### Byte-Pair Encoding (BPE)

BPE starts with a character vocabulary and iteratively merges the most frequent adjacent pair in the training corpus. At each iteration, the algorithm finds the pair $(a, b)$ with maximum frequency and adds a new symbol $ab$ to the vocabulary:

$$\text{merge}^{(t)} = \arg\max_{(x,y) \in \mathcal{P}^{(t)}} \text{freq}(x, y)$$

where $\mathcal{P}^{(t)}$ is the set of all adjacent symbol pairs at iteration $t$. This greedy process continues until a target vocabulary size is reached. BPE was popularized by GPT-2 and is deterministic at inference time.

### WordPiece

WordPiece (used by BERT) is similar but selects merges based on **likelihood gain** rather than raw frequency:

$$\text{merge}^{(t)} = \arg\max_{(x,y)} \frac{\text{freq}(xy)}{\text{freq}(x)\text{freq}(y)}$$

This ratio measures how much adding the merged token $xy$ increases the corpus likelihood. It tends to produce more linguistically meaningful subwords than pure frequency-based BPE.

### Unigram Language Model

Unigram (used by T5, XLNet) takes the opposite approach: start with a large seed vocabulary and **prune** tokens that least reduce corpus likelihood. The training objective is to find a vocabulary $\mathcal{V}$ that maximizes:

$$\mathcal{L}(\mathcal{V}) = \sum_{s \in \mathcal{D}} \log \sum_{\mathbf{x} \in \mathcal{S}(s)} \prod_{i=1}^{|\mathbf{x}|} P(x_i)$$

where $\mathcal{S}(s)$ is the set of all segmentations of sequence $s$. This probabilistic formulation enables **subword regularization** (randomly sampling different segmentations during training), which improves robustness.

### Byte-Level BPE (BBPE)

BBPE operates on UTF-8 bytes rather than Unicode characters, guaranteeing that **every** input string is representable. This is the backbone of GPT-2, RoBERTa, and LLaMA models. The trade-off is a slightly larger vocabulary (256 byte tokens + merge tokens) and less interpretable subword units, but zero out-of-vocabulary tokens.

### The tokenizers Library

The `tokenizers` library implements these algorithms in Rust with Python bindings, achieving 10-100x speedups over pure Python tokenizers. The processing pipeline has four stages:

```python
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Fast Rust-backed tokenizer (always prefer use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
print(f"Fast: {tokenizer.is_fast}")  # True
print(f"Vocab size: {tokenizer.vocab_size}")

# Encode a single string
encoded = tokenizer.encode_plus(
    "HuggingFace is great!",
    add_special_tokens=True,
    max_length=12,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
print(encoded["input_ids"])       # tensor([[101, 17662, ..., 102]])
print(encoded["attention_mask"])  # tensor([[1, 1, ..., 0]])

# Batch encoding with dynamic padding
batch = tokenizer.batch_encode_plus(
    ["Short.", "A much longer sentence here for testing."],
    padding="longest",
    truncation=True,
    return_tensors="pt"
)
print(batch["input_ids"].shape)  # (2, 9) — padded to longest in batch
```

### Training a Custom Tokenizer

Essential for domain-specific text (genomic sequences, source code, legal documents):

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

tokenizer_new = Tokenizer(models.BPE())
tokenizer_new.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2
)

files = ["corpus.txt"]
tokenizer_new.train(files, trainer)
tokenizer_new.save("custom_tokenizer.json")

# Load and use
from tokenizers import Tokenizer
t = Tokenizer.from_file("custom_tokenizer.json")
output = t.encode("Hello world")
print(output.ids)
```

✅ **Antipattern: Ignoring the fast tokenizer**
```python
# ❌ Might load slow Python tokenizer depending on model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ Explicitly request fast Rust-backed version
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
```

> **Caso real: GitHub Copilot** uses a Byte-level BPE tokenizer trained on source code. Because code contains rare identifiers (variable names, library paths, Unicode string literals), BBPE guarantees zero `<unk>` tokens. The Rust tokenizer achieves <10ms per request in high-concurrency serving.

⚠️ **Left-side vs right-side truncation**: BERT truncates from the right by default (`truncation_side="right"`). For tasks where conclusions appear at the end of a document, this silently drops the most important tokens. Set `truncation_side="left"` when the end is critical.

💡 **Smoke test**: Always verify `tokenizer.vocab_size == model.config.vocab_size` to catch mismatched tokenizer/model pairs that produce silent gibberish.

---

## 2. The datasets Library and Arrow Backend

Modern NLP datasets exceed RAM capacity. The `datasets` library, built on Apache Arrow, provides memory-mapped, columnar storage that enables terabyte-scale data processing without exhausting system memory. Arrow's zero-copy semantics mean slicing and batching do not duplicate data.

### Core Operations

```python
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load from Hub or local files (CSV, JSON, Parquet)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Streaming for web-scale data (no full download)
stream = load_dataset(
    "oscar", "unsharded_deduplicated_en",
    split="train",
    streaming=True
)

# Transform: map applies a function lazily or eagerly
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,          # 10x+ faster than single examples
    num_proc=4,            # Parallelize across CPU cores
    remove_columns=["text"]  # Drop raw text to save memory
)

# Filter to remove outliers
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)

# Mix multiple data sources
mixed = interleave_datasets(
    [dataset_a, dataset_b],
    probabilities=[0.7, 0.3]
)

# Materialize to disk for reuse across training runs
tokenized.save_to_disk("./tokenized_wikitext")
# Later: from datasets import load_from_disk
```

### Streaming Considerations

Streaming datasets (`streaming=True`) fetch rows on demand, enabling training on corpora larger than local disk. However, streaming disables random-access shuffling. Instead, `dataset.shuffle()` uses a fixed-size buffer:

```python
# Approximate shuffle with buffer
shuffled = stream.shuffle(buffer_size=100_000)

# No len(), no indexing, no repeat()
for batch in shuffled:
    pass
```

If your data has strong ordering bias (all Wikipedia before all code), a small buffer cannot global-shuffle, which can hurt convergence.

✅ **Antipattern: Memory explosion with map**
```python
# ❌ batched=False + no remove_columns = materialize all columns + Python objects
dataset.map(tokenize_function, batched=False)

# ✅ batched=True + remove_columns = lean Arrow pipeline
dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

> **Caso real: MosaicML** uses `streaming=True` with `interleave_datasets` to compose training mixtures for MPT models. Their data platform streams sharded Arrow files from object storage, applies on-the-fly tokenization, and feeds directly into `Trainer` without ever materializing the full dataset on local NVMe.

⚠️ **num_proc concurrency**: `dataset.map(num_proc=8)` speeds tokenization via multiprocessing, but PyTorch `DataLoader(num_workers=4)` is a different concern (parallel batch loading). Higher `num_proc` does not always help if the tokenizer is already I/O bound.

---

## 3. DataCollators and Padding Strategies

`DataCollator` objects bridge variable-length tokenized examples and fixed-shape tensors required by PyTorch. Sequences within a batch differ in length, so the collator pads them to a common size and constructs an attention mask so the model ignores padding positions.

### Padding Efficiency

Given a batch of $N$ sequences with lengths $\ell_1, \ell_2, \dots, \ell_N$, the padding waste for a global max length $L$ is:

$$W_{\text{global}} = \sum_{i=1}^N (L - \ell_i)$$

With dynamic padding to the batch maximum $\hat{\ell} = \max(\ell_1, \dots, \ell_N)$:

$$W_{\text{dynamic}} = \sum_{i=1}^N (\hat{\ell} - \ell_i)$$

Dynamic padding is always at least as efficient as global padding, and typically much more so for variable-length data.

```python
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader

# Dynamic padding to longest in each batch
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

loader = DataLoader(
    tokenized,
    batch_size=16,
    collate_fn=collator,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    # batch["input_ids"].shape = (16, max_len_in_batch)
    # batch["attention_mask"].shape = (16, max_len_in_batch)
    break
```

### Task-Specific Collators

```python
# Masked Language Modeling (BERT-style)
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Causal Language Modeling (GPT-style)
clm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Seq2Seq (T5, BART) — requires shifted decoder inputs
# seq2seq_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model
# )
```

### The Attention Mask

The attention mask is a binary tensor where $M_{ij} = 1$ for real tokens and $M_{ij} = 0$ for padding. Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + (1 - M) \cdot (-\infty)\right) V$$

Setting attention to $-\infty$ for padding positions ensures they contribute zero to the weighted sum. This is why the mask is mandatory for batched inference with variable-length sequences.

✅ **Antipattern: Manual padding without masks**
```python
# ❌ Naive padding breaks self-attention
# Manually pad to 512 without attention mask

# ✅ DataCollator handles padding and masks automatically
collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

⚠️ **remove_unused_columns=True (default)**: If your dataset has extra columns needed in `compute_metrics`, `Trainer` silently drops them. Set `remove_unused_columns=False` for custom evaluation.

💡 **Materialization**: Use `dataset.save_to_disk()` after tokenization to avoid recomputing transforms across training restarts. This eliminates non-determinism and saves time for large teams.

## 🎯 Key Takeaways

- Subword tokenization (BPE, WordPiece, Unigram, BBPE) bridges infinite text and finite vocabulary; BBPE guarantees zero `<unk>` tokens.
- The `tokenizers` library provides Rust-backed speed; always prefer `AutoTokenizer` with `use_fast=True`.
- Padding strategy matters: `padding="longest"` saves memory per batch, `padding="max_length"` simplifies shape logic.
- The `datasets` library uses Apache Arrow for memory-efficient, lazy, and parallelizable data processing.
- `DataCollator` objects dynamically align variable-length sequences into fixed-shape tensors with correct attention masks.
- Streaming datasets enable training on web-scale corpora but sacrifice random access and exact shuffling.
- `DataCollatorForLanguageModeling` handles both MLM and CLM with automatic label construction.
- Materializing preprocessed datasets with `save_to_disk` eliminates recomputation and non-determinism.
- Always verify `tokenizer.is_fast` is `True` in production; the Python fallback is orders of magnitude slower.

## References

- Hugging Face Tokenizers Docs: [https://huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers)
- Hugging Face Datasets Docs: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", ACL 2016.
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer", EMNLP 2018.
- Related Vault: [[01 - The from_pretrained Ecosystem]]
- Related Vault: [[03 - Trainer, TrainingArguments, and Distributed Training]]

## Código de compresión

```python
"""
End-to-end data pipeline: load, tokenize, filter, collate, and iterate.
"""
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

TOKENIZER = "bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
book = load_dataset("bookcorpus", split="train")
mixed = interleave_datasets([wiki, book], probabilities=[0.5, 0.5])

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

tokenized = mixed.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=mixed.column_names
)

tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)

collator = DataCollatorWithPadding(tokenizer=tokenizer)
loader = DataLoader(tokenized, batch_size=BATCH_SIZE, collate_fn=collator)

for batch in loader:
    print(f"Batch shape: {batch['input_ids'].shape}, "
          f"Mask shape: {batch['attention_mask'].shape}")
    break
```
