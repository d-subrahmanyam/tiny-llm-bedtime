# Tiny LLM from Bedtime Stories

Learning-first project to build a GPT-style language model end-to-end using a Hugging Face bedtime-stories dataset.

## What you get

- Trainable tiny Transformer model (`TinyGPT`) in PyTorch
- Custom BPE tokenizer training + loading
- CLI inference (`tiny-llm-generate`)
- FastAPI server (`tiny-llm-api`)
- Concept docs + UML diagrams for study
- Checkpoint generation pipeline (`checkpoints/*.pt`), kept out of git due GitHub file-size limits

## Quick start (uv)

### 1) Install uv (one-time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2) Install dependencies

```bash
uv sync
```

### 3) Train

```bash
uv run tiny-llm-train --config configs/default.yaml
```

Artifacts:
- tokenizer: `tokenizer.json`
- checkpoints: [`checkpoints/`](./checkpoints)

> Note: checkpoint `.pt` files are intentionally ignored in git (GitHub blocks files over 100MB).  
> Train locally to generate `checkpoints/step_005000.pt` before running inference/API commands below.

### 4) Generate from a checkpoint

```bash
uv run tiny-llm-generate \
  --config configs/default.yaml \
  --checkpoint checkpoints/step_005000.pt \
  --prompt "Once upon a moonlit night," \
  --max-new-tokens 120 \
  --temperature 0.7 \
  --top-k 20
```

### 5) Run the API

```bash
uv run tiny-llm-api \
  --config configs/default.yaml \
  --checkpoint checkpoints/step_005000.pt \
  --host 127.0.0.1 \
  --port 8000
```

Generate via API:

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a moonlit night,",
    "max_new_tokens": 120,
    "temperature": 0.7,
    "top_k": 20
  }'
```

Interactive docs: `http://127.0.0.1:8000/docs`

## Download model from GitHub Releases

Release: `v0.1.0`  
URL: `https://github.com/d-subrahmanyam/tiny-llm-bedtime/releases/tag/v0.1.0`

### Option A: with GitHub CLI

```bash
gh release download v0.1.0 \
  -R d-subrahmanyam/tiny-llm-bedtime \
  -p "step_005000.pt" \
  -p "tokenizer.json" \
  -D .
mkdir -p checkpoints
mv step_005000.pt checkpoints/step_005000.pt
```

### Option B: with curl

```bash
curl -L -o checkpoints/step_005000.pt \
  https://github.com/d-subrahmanyam/tiny-llm-bedtime/releases/download/v0.1.0/step_005000.pt
curl -L -o tokenizer.json \
  https://github.com/d-subrahmanyam/tiny-llm-bedtime/releases/download/v0.1.0/tokenizer.json
```

## Config

Main config: [`configs/default.yaml`](./configs/default.yaml)

Important knobs:
- `data.dataset_name`, `data.text_field`
- `model.n_layer`, `model.n_embd`, `model.block_size`
- `training.max_steps`, `training.learning_rate`, `training.batch_size`

## Project layout

- [`src/tiny_llm/train.py`](./src/tiny_llm/train.py): training loop + checkpointing
- [`src/tiny_llm/model.py`](./src/tiny_llm/model.py): Transformer implementation
- [`src/tiny_llm/data.py`](./src/tiny_llm/data.py): dataset/tokenizer/batches
- [`src/tiny_llm/inference.py`](./src/tiny_llm/inference.py): checkpoint + tokenizer loading for inference
- [`src/tiny_llm/generate.py`](./src/tiny_llm/generate.py): CLI generation
- [`src/tiny_llm/api.py`](./src/tiny_llm/api.py): FastAPI app
- [`docs/`](./docs): learning notes and diagrams

## Learning docs

- [`docs/01_overview.md`](./docs/01_overview.md)
- [`docs/02_data_and_tokenization.md`](./docs/02_data_and_tokenization.md)
- [`docs/03_transformer_architecture.md`](./docs/03_transformer_architecture.md)
- [`docs/04_training_loop.md`](./docs/04_training_loop.md)
- [`docs/05_inference_and_sampling.md`](./docs/05_inference_and_sampling.md)
- [`docs/06_scaling_and_next_steps.md`](./docs/06_scaling_and_next_steps.md)
- [`docs/07_serving_api.md`](./docs/07_serving_api.md)
- [`docs/uml_diagrams.md`](./docs/uml_diagrams.md)

## Notes

- This is an educational tiny model; coherence is limited by data and model size.
- For better outputs, increase dataset size and training budget.
