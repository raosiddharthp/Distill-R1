# Distill-R1

**Domain-adapted LLM knowledge distillation framework for enterprise inference at scale.**

Transfer reasoning capability from a large open-source teacher model into a small, fast, domain-expert student — fully on-premises, legally clean, 25× cheaper than proprietary APIs at production volume.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Teacher: Llama 3.3 70B](https://img.shields.io/badge/Teacher-Llama%203.3%2070B-orange.svg)](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
[![Student: Qwen2.5 7B](https://img.shields.io/badge/Student-Qwen2.5%207B-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Stack: Vertex AI](https://img.shields.io/badge/Stack-Vertex%20AI-4285F4.svg)](https://cloud.google.com/vertex-ai)

---

## Table of Contents

- [Overview](#overview)
- [Why Distill-R1](#why-distill-r1)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [Phase 1: Synthetic Data Generation](#phase-1-synthetic-data-generation)
  - [Phase 2: QLoRA Fine-Tuning](#phase-2-qlora-fine-tuning)
  - [Phase 3: Evaluation and Adapter Merge](#phase-3-evaluation-and-adapter-merge)
- [Configuration](#configuration)
- [Deployment](#deployment)
  - [Free-Tier Demo Path](#free-tier-demo-path)
  - [GCP Production Path](#gcp-production-path)
- [Evaluation](#evaluation)
- [Cost Model](#cost-model)
- [Legal and Compliance](#legal-and-compliance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Distill-R1 is a reference implementation of a three-phase LLM knowledge distillation pipeline targeting enterprise domain adaptation. It demonstrates how to:

1. Generate a high-quality synthetic training dataset using an open-source teacher model (Llama 3.3 70B)
2. Fine-tune a small student model (Qwen2.5 7B) using QLoRA on a single GPU
3. Evaluate, merge, and serve the resulting domain-expert model via GCP Vertex AI

The worked example throughout is a **retail product intelligence assistant** — a model that answers queries about 80,000+ SKUs, return policies, stock availability, and product comparisons. The pattern applies to any high-volume, domain-specific enterprise use case.

| Metric | Value |
|--------|-------|
| Inference cost reduction vs. GPT-4o | **25×** at 10M queries/day |
| Task accuracy retained vs. teacher | **~94%** on domain eval set |
| Training cost (Vertex AI A100 × 4hrs) | **~$50** one-time |
| Portfolio demo path cost | **$0** (Kaggle + HF Spaces) |
| ToS legal exposure | **$0** (Llama 3.1+ licence permits distillation) |
| Serving latency p50 | **~80ms** (vLLM + PagedAttention on T4) |

---

## Why Distill-R1

Proprietary LLM APIs carry three compounding failure modes at production scale:

**Cost.** At 10M queries/day with ~500 tokens average, GPT-4o costs ~$50,000/day. Claude Sonnet costs ~$37,500/day. A domain-fine-tuned Qwen2.5 7B on Vertex AI preemptible T4s costs ~$200/day. Training cost (~$50) is recovered within the first hour of production deployment.

**Data sovereignty.** Every query sent to a proprietary API exits the enterprise network. For retail, healthcare, finance, or legal workloads, this creates GDPR, PDPA, and SOC 2 exposure on every single inference call.

**Domain fit.** Base models are trained on the internet, not your SKU catalogue, return policy §4.2, or internal pricing logic. They hallucinate confidently on domain-specific queries. Fine-tuning bakes domain knowledge into the weights — no retrieval infrastructure required for straightforward factual lookups.

Distill-R1 solves all three simultaneously. The teacher model is **Llama 3.3 70B** — whose 3.1+ licence explicitly permits using its outputs to train other models. No proprietary model output is used at any stage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING PLANE  (run once per model version)                       │
│                                                                     │
│  Domain Docs                                                        │
│  (PDF/JSON)  ──▶  Llama 3.3 70B  ──▶  Synthetic Dataset            │
│                   (Teacher)            (500–2k JSONL)               │
│                                             │                       │
│                                             ▼                       │
│                                       QLoRA Trainer                 │
│                                       Qwen2.5 7B frozen             │
│                                       rank=16 · NF4 4-bit           │
│                                             │                       │
│                                             ▼                       │
│                                       LoRA Adapters (~80MB)         │
│                                             │                       │
│                            ┌── eval pass ───┤                       │
│                            │                ▼                       │
│                            │          Merge + Push                  │
│                            │          HF Hub / Model Registry       │
│                            │                                        │
│                            └── eval fail ──▶ retrain trigger        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SERVING PLANE  (always-on)                                         │
│                                                                     │
│  Client  ──▶  Cloud Armor  ──▶  Global LB  ──▶  Cloud Run          │
│               (WAF/DDoS)        (CDN)           (API gateway)       │
│                                                       │             │
│                                                       ▼             │
│                                              Vertex AI Prediction   │
│                                              vLLM · PagedAttention  │
│                                              T4×1 · autoscale 1→8   │
│                                              ~80ms p50              │
│                                                       │             │
│                                                       ▼             │
│                                              Model Registry         │
└─────────────────────────────────────────────────────────────────────┘
```

The two planes share only the **Model Registry** as their interface. This enables independent scaling, CI/CD, and failure domains.

---

## Prerequisites

### Local / Free-Tier Training (Kaggle / Colab)

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| CUDA | 11.8+ |
| GPU VRAM | 16 GB minimum (T4 sufficient for Qwen2.5 7B QLoRA) |
| `transformers` | 4.43.0+ |
| `peft` | 0.11.0+ |
| `trl` | 0.8.0+ |
| `unsloth` | latest |
| `bitsandbytes` | 0.43.0+ |

### GCP Production Path

- GCP project with billing enabled
- Vertex AI API enabled (`gcloud services enable aiplatform.googleapis.com`)
- GPU quota: at least 1× A100 40GB in `asia-southeast1` (or your target region)
- `gcloud` CLI authenticated
- Cloud Storage bucket for data and artifacts
- OpenRouter API key (for teacher inference — free tier sufficient for dataset generation)

> **Billing note for training experiments:** Use a dedicated Google account with no payment method attached for Kaggle-based training. Structural billing risk is zero. For GCP, set a budget alert at $20 before requesting GPU quota.

---

## Quick Start

```bash
git clone https://github.com/raosiddharthp/distill-r1.git
cd distill-r1
pip install -r requirements.txt
```

### 1. Configure your environment

```bash
cp config/example.yaml config/local.yaml
# Edit config/local.yaml with your OpenRouter API key,
# GCS bucket name, and target domain settings
```

### 2. Generate synthetic training data

```bash
python pipeline/01_generate.py \
  --config config/local.yaml \
  --input data/raw/product_catalogue.pdf \
  --output data/synthetic/train.jsonl \
  --num_samples 1000
```

### 3. Run QLoRA fine-tuning (local T4)

```bash
python pipeline/02_finetune.py \
  --config config/local.yaml \
  --dataset data/synthetic/train.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --output_dir checkpoints/retail-v1
```

### 4. Evaluate and merge adapters

```bash
python pipeline/03_evaluate.py \
  --config config/local.yaml \
  --adapter_path checkpoints/retail-v1 \
  --test_set data/eval/test.jsonl

python pipeline/04_merge.py \
  --adapter_path checkpoints/retail-v1 \
  --output_dir models/retail-v1-merged \
  --push_to_hub  # optional: push to HF Hub
```

---

## Project Structure

```
distill-r1/
├── config/
│   ├── example.yaml          # Configuration template
│   └── vertex_pipeline.yaml  # Vertex AI Pipelines definition
├── pipeline/
│   ├── 01_generate.py        # Synthetic data generation (teacher)
│   ├── 02_finetune.py        # QLoRA fine-tuning (student)
│   ├── 03_evaluate.py        # EM / ROUGE-L / LLM-as-judge eval
│   └── 04_merge.py           # Adapter merge + HF Hub push
├── serving/
│   ├── cloud_run/            # Cloud Run API gateway (FastAPI)
│   └── vertex_prediction/    # vLLM serving container
├── infra/
│   ├── terraform/            # GCP infrastructure as code
│   └── vertex_pipeline.py    # Kubeflow pipeline definition
├── demo/
│   └── app.py                # HF Spaces Gradio demo (pre-computed)
├── notebooks/
│   ├── kaggle_finetune.ipynb # Full training walkthrough (Kaggle T4)
│   └── colab_quickstart.ipynb
├── data/
│   ├── raw/                  # Input domain documents
│   ├── synthetic/            # Generated training data
│   └── eval/                 # Held-out evaluation set
├── tests/
├── requirements.txt
└── README.md
```

---

## Pipeline

### Phase 1: Synthetic Data Generation

The teacher model (Llama 3.3 70B via OpenRouter) ingests raw domain documents and generates structured Q&A pairs using chain-of-thought prompting. Chain-of-thought is critical — it forces the teacher to expose its reasoning process, not just its answers, which gives the student richer training signal.

**Prompt template:**

```
You are a domain expert assistant. Given the following product documentation,
generate a realistic customer query and a detailed, accurate answer that
includes relevant policy citations, SKU details, and actionable next steps.

Document:
{document_chunk}

Generate a JSON object with keys: "query", "response", "reasoning_steps".
The response must cite specific sections, SKU numbers, or policy clauses
exactly as they appear in the document.
```

**Output format (Alpaca/ShareGPT):**

```jsonl
{"instruction": "Does SKU #44821 ship to Singapore?", "output": "SKU #44821 (KALLAX 2×4, birch effect) ships to Singapore via standard delivery..."}
```

**Cost:** ~$0 for 1,000 samples at Llama 3.3 70B free tier on OpenRouter.

---

### Phase 2: QLoRA Fine-Tuning

Qwen2.5 7B is loaded in 4-bit NF4 precision via `bitsandbytes`, then Low-Rank Adaptation adapters are attached to the attention projection matrices. Only the adapter weights (~0.3% of total parameters) are updated during training. The frozen base model handles general language; the adapters learn domain-specific reasoning patterns.

**Key hyperparameters:**

```python
# LoRA config
lora_config = LoraConfig(
    r=16,                        # rank — higher = more capacity, more VRAM
    lora_alpha=32,               # scaling factor (alpha/r = 2 is standard)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Quantisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",   # NormalFloat4 — optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantisation saves ~0.4 GB extra
)

# Training args
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)
```

**VRAM requirements:**

| Configuration | VRAM Required |
|---------------|--------------|
| Full fine-tune Qwen2.5 7B | ~120 GB |
| QLoRA rank=16, NF4 4-bit | ~16 GB |
| QLoRA rank=16, NF4 4-bit + Unsloth | ~12 GB |

Unsloth's custom CUDA kernels reduce memory by a further 25–40% and increase training throughput by ~2×. Recommended for all single-GPU training runs.

---

### Phase 3: Evaluation and Adapter Merge

Three evaluation metrics are computed on the held-out test set before adapters are promoted:

| Metric | Target | Description |
|--------|--------|-------------|
| **Exact Match (EM)** | ≥ 0.85 | For factual queries with deterministic answers (SKU numbers, policy sections, prices) |
| **ROUGE-L** | ≥ 0.72 | For generative responses requiring coverage of key information |
| **LLM-as-judge** | ≥ 4.0 / 5.0 | Teacher (Llama 3.3 70B) scores student responses on accuracy, completeness, and citation quality |

If all three pass the configured thresholds, adapters are merged into the base model weights via `peft.merge_and_unload()`. This produces standard `safetensors` files with zero inference latency overhead — the merged model is indistinguishable from a standard Hugging Face model and can be served directly with vLLM.

```python
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained("models/retail-v1-merged")
tokenizer.save_pretrained("models/retail-v1-merged")
```

---

## Configuration

All pipeline behaviour is controlled via a single YAML config file:

```yaml
# config/example.yaml

project:
  name: distill-r1-retail
  domain: retail_product_intelligence
  version: "1.0.0"

teacher:
  model: meta-llama/llama-3.3-70b-instruct:free
  provider: openrouter
  api_key: ${OPENROUTER_API_KEY}
  temperature: 0.7
  max_tokens: 1024
  num_samples: 1000

student:
  base_model: Qwen/Qwen2.5-7B-Instruct
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  quantisation: nf4
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  bf16: true
  use_unsloth: true

evaluation:
  exact_match_threshold: 0.85
  rouge_l_threshold: 0.72
  llm_judge_threshold: 4.0
  judge_model: meta-llama/llama-3.3-70b-instruct:free

gcp:
  project_id: ${GCP_PROJECT_ID}
  region: asia-southeast1
  bucket: gs://distill-r1-data
  artifact_registry: asia-southeast1-docker.pkg.dev/${GCP_PROJECT_ID}/distill-r1
  machine_type: n1-standard-8
  accelerator_type: NVIDIA_TESLA_A100
  accelerator_count: 1

serving:
  min_replicas: 1
  max_replicas: 8
  target_utilisation: 0.7
  machine_type: n1-standard-4
  accelerator_type: NVIDIA_TESLA_T4
  accelerator_count: 1
  use_preemptible: true
```

---

## Deployment

### Free-Tier Demo Path

The full pipeline runs to completion at zero cost using only free-tier services:

```
Kaggle T4 (train)  →  HF Hub (store)  →  HF Spaces CPU (demo)
```

1. **Training:** Run `notebooks/kaggle_finetune.ipynb` on Kaggle (30 GPU hrs/week free, no payment method required). Training Qwen2.5 7B QLoRA on 1,000 samples takes approximately 3–4 hours on a T4.

2. **Storage:** Push merged weights to a private HF Hub repository. Free for models under 10 GB.

3. **Demo:** Deploy `demo/app.py` to HF Spaces (CPU tier, always-on, no cold starts). The demo uses pre-computed inference results stored as JSON — no GPU required at serve time, loads in under 300ms.

### GCP Production Path

The production path uses Vertex AI Pipelines for orchestration. Each pipeline run executes a 7-step Kubeflow DAG: Ingest → Synth Gen → QLoRA Train → Evaluate → [gate] → Register → Push Hub → Notify.

**Deploy infrastructure:**

```bash
cd infra/terraform
terraform init
terraform apply -var="project_id=$GCP_PROJECT_ID"
```

**Trigger a training pipeline run:**

```bash
python infra/vertex_pipeline.py \
  --config config/vertex_pipeline.yaml \
  --data_uri gs://distill-r1-data/raw/catalogue_v2/ \
  --experiment_name retail-v2
```

**Deploy serving endpoint:**

```bash
# Build and push vLLM serving container
cd serving/vertex_prediction
gcloud builds submit --tag $ARTIFACT_REGISTRY/vllm-qwen:latest

# Create Vertex AI endpoint
gcloud ai endpoints create \
  --region=asia-southeast1 \
  --display-name=distill-r1-retail

# Deploy model to endpoint
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=asia-southeast1 \
  --model=$MODEL_ID \
  --machine-type=n1-standard-4 \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --min-replica-count=1 \
  --max-replica-count=8
```

**Deploy Cloud Run API gateway:**

```bash
cd serving/cloud_run
gcloud run deploy distill-r1-api \
  --source . \
  --region asia-southeast1 \
  --set-env-vars VERTEX_ENDPOINT_ID=$ENDPOINT_ID \
  --allow-unauthenticated
```

> **GPU quota:** New GCP accounts have conservative GPU quotas. Submit a quota increase request for `NVIDIA_TESLA_A100` and `NVIDIA_TESLA_T4` in your target region before running training. Approval typically takes 1–2 business days.

---

## Evaluation

Run the full evaluation suite against any model checkpoint:

```bash
python pipeline/03_evaluate.py \
  --adapter_path checkpoints/retail-v1 \
  --test_set data/eval/test.jsonl \
  --judge_model meta-llama/llama-3.3-70b-instruct:free \
  --output_dir results/retail-v1/
```

Sample output:

```
Evaluation Results — retail-v1
──────────────────────────────────────────
Exact Match (EM):       0.912   [PASS ≥ 0.85]
ROUGE-L:                0.784   [PASS ≥ 0.72]
LLM-as-judge (avg):     4.3/5   [PASS ≥ 4.0]

Per-category breakdown:
  sku_lookup:           EM=0.97   ROUGE=0.81
  policy_query:         EM=0.94   ROUGE=0.78
  product_comparison:   EM=0.88   ROUGE=0.76
  availability:         EM=0.89   ROUGE=0.79

Adapter promotion: APPROVED
──────────────────────────────────────────
```

---

## Cost Model

| Deployment pattern | Inference cost | Data sovereignty | Domain fit | Recommended for |
|--------------------|---------------|-----------------|------------|-----------------|
| Proprietary API (GPT-4o) | ~$50,000/day | ✗ None | ✗ Prompt only | Prototyping |
| Proprietary API (Claude) | ~$37,500/day | ✗ None | ✗ Prompt only | Prototyping |
| OSS API (Together AI) | ~$2,000/day | ⚠ 3rd party | ⚠ RAG + prompt | Low-volume |
| Self-hosted Llama 70B | ~$600/day | ✓ Full | ⚠ Prompt + RAG | Privacy-first |
| **Distill-R1 (Qwen2.5 7B)** | **~$200/day** | **✓ Full** | **✓ Baked in** | **High-volume, domain-specific** |

Assumes 10M queries/day · ~500 tokens average · Vertex AI preemptible T4 pricing · excludes storage and egress. Break-even on training cost (~$50) occurs within the first 3 minutes of production operation at this query volume.

---

## Legal and Compliance

**This framework exclusively uses Llama 3.3 70B as the teacher model.** Meta's Llama 3.1+ licence explicitly permits using model outputs to train other models:

> *"We're making changes to our licence, allowing developers to use the outputs from Llama models — including the 405B — to improve other models."*  
> — Meta AI, Llama 3.1 Release, July 2024

**Do not substitute a proprietary model (Claude, GPT-4o, Gemini) as the teacher.** Anthropic, OpenAI, and Google explicitly prohibit using their model outputs for training competing models in their Terms of Service. Anthropic publicly identified and named three AI companies for exactly this practice in February 2026.

The student model (Qwen2.5 7B) is released under Apache 2.0, which permits commercial use, modification, and redistribution without restriction.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for any change beyond a bug fix.

```bash
git clone https://github.com/raosiddharthp/distill-r1.git
cd distill-r1
pip install -r requirements-dev.txt
pre-commit install
pytest tests/
```

Areas where contributions are most useful:
- Additional domain adapters (healthcare, legal, financial services)
- Alternative student models (Phi-3, Mistral, Gemma 2)
- GKE-based serving path as an alternative to Vertex AI Prediction
- Automated hyperparameter sweep integration (Vertex AI Vizier)
- GGUF quantisation for edge/on-prem deployment

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The base models (Llama 3.3 70B, Qwen2.5 7B) are governed by their respective upstream licences. Trained adapter weights derived from this pipeline inherit the student model's Apache 2.0 licence.

---

*Distill-R1 · Architecture Portfolio · 2026 · [raosiddharthp.github.io](https://raosiddharthp.github.io)*
