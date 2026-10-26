# 1) Tiny LLM Overview

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

## Core artifacts

- `tokenizer.json`: mapping between text pieces and token IDs.
- `checkpoints/*.pt`: model snapshots during training.
- Config file: single source of truth for hyperparameters.

