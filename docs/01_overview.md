# 1) Tiny LLM Overview

## Pipeline at a glance

```text
Raw stories (HF dataset)
        |
        v
[1] Load + clean text rows
        |
        v
[2] Train/load BPE tokenizer ---------------> tokenizer.json
        |
        v
[3] Encode corpus + add <|eos|> boundaries
        |
        v
[4] Split into train_tokens / val_tokens
        |
        v
[5] Sample random x,y batches (next-token targets)
        |
        v
[6] TinyGPT forward pass
    embeddings -> masked self-attention -> MLP -> logits
        |
        v
[7] Cross-entropy loss + AdamW update
        |
        v
[8] Periodic eval + checkpointing ----------> checkpoints/*.pt
        |
        v
[9] Inference sampling (CLI) OR [10] API serving (FastAPI)
```

## What is an LLM?

A language model learns a probability distribution over token sequences:

\[
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
\]

In plain English: it predicts the next token from previous tokens.

## Why build a tiny one first?

- Fast experimentation.
- Easier debugging.
- Better intuition before scaling.

## End-to-end pipeline in this project

1. Load bedtime-style stories from Hugging Face.
2. Train a tokenizer (subword BPE).
3. Convert text into token IDs.
4. Train a causal Transformer with next-token loss.
5. Sample stories from the trained model.

## Stage-by-stage builder journey

Use this as a practical map while reading the codebase.

| Stage | What happens | Input -> Output | Primary code/docs |
|---|---|---|---|
| 1. Data load | Pull stories from Hugging Face and filter valid text rows. | Dataset rows -> `list[str]` texts | `src/tiny_llm/data.py` (`load_texts`), [02_data_and_tokenization.md](./02_data_and_tokenization.md) |
| 2. Tokenizer | Train (or load) byte-level BPE with special tokens and persist it. | Texts -> `tokenizer.json` | `src/tiny_llm/data.py` (`train_or_load_tokenizer`), [02_data_and_tokenization.md](./02_data_and_tokenization.md) |
| 3. Token stream | Encode each story to token IDs and append `<\|eos\|>` between stories. | Texts + tokenizer -> `torch.Tensor` token stream | `src/tiny_llm/data.py` (`encode_corpus`), [02_data_and_tokenization.md](./02_data_and_tokenization.md) |
| 4. Train/val split | Split one long stream into train and validation partitions. | Token stream -> `train_tokens`, `val_tokens` | `src/tiny_llm/data.py` (`split_train_val`) |
| 5. Batch sampling | Randomly sample contiguous windows and build shifted targets. | `train_tokens` -> `x, y` batches | `src/tiny_llm/data.py` (`get_batch`), [04_training_loop.md](./04_training_loop.md) |
| 6. Transformer forward | Embeddings -> masked attention -> MLP blocks -> logits over vocab. | `x` token IDs -> logits | `src/tiny_llm/model.py`, [03_transformer_architecture.md](./03_transformer_architecture.md) |
| 7. Loss + optimize | Compute next-token cross-entropy, backprop, clip gradients, AdamW step. | logits + `y` -> updated weights | `src/tiny_llm/train.py`, [04_training_loop.md](./04_training_loop.md) |
| 8. Eval + checkpoint | Periodically estimate train/val loss, preview text, and save `.pt`. | model state -> metrics + `checkpoints/*.pt` | `src/tiny_llm/train.py` |
| 9. Inference sampling | Load checkpoint + tokenizer, then autoregressively sample next tokens. | prompt text -> generated text | `src/tiny_llm/inference.py`, `src/tiny_llm/generate.py`, [05_inference_and_sampling.md](./05_inference_and_sampling.md) |
| 10. API serving | Expose generation via HTTP endpoint for app integration. | JSON request -> JSON response | `src/tiny_llm/api.py`, [07_serving_api.md](./07_serving_api.md) |

## What to run at each stage

1. Install deps: `uv sync`
2. Build artifacts (tokenizer/checkpoints) via training:
   `uv run tiny-llm-train --config configs/default.yaml`
3. Test text generation:
   `uv run tiny-llm-generate --config configs/default.yaml --checkpoint checkpoints/step_005000.pt --prompt "Once upon a moonlit night,"`
4. Serve over HTTP:
   `uv run tiny-llm-api --config configs/default.yaml --checkpoint checkpoints/step_005000.pt`

## Core artifacts

- `tokenizer.json`: mapping between text pieces and token IDs.
- `checkpoints/*.pt`: model snapshots during training.
- Config file: single source of truth for hyperparameters.
