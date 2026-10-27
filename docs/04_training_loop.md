# 4) Training Loop

Training code is in [train.py](/home/subbu/Code/try-llm/tiny-llm/src/tiny_llm/train.py).

## Objective

Predict next token for each position.

If input chunk is:

`[x1, x2, x3, x4]`

targets are:

`[x2, x3, x4, x5]`

Loss = cross-entropy between predicted logits and target IDs.

## Batch construction

Random start positions are sampled from one long token stream.
Each batch returns:

- `x`: `[batch_size, block_size]`
- `y`: shifted by one position, same shape

## Optimization

- Optimizer: AdamW
- Gradient clipping: improves stability
- Periodic evaluation: train/val average loss
- Checkpointing: saves model and optimizer states

## How to read loss values

- Lower loss usually means better next-token prediction.
- If train loss drops but val loss stagnates or rises, overfitting is likely.

## Config chosen in this project (and why)

From [configs/default.yaml](/home/subbu/Code/try-llm/tiny-llm/configs/default.yaml):

- `batch_size: 16`
- `learning_rate: 0.00025`
- `weight_decay: 0.1`
- `betas: [0.9, 0.95]`
- `max_steps: 5000`
- `eval_interval: 200`, `eval_iters: 50`
- `gradient_clip: 1.0`
- `checkpoint_every: 1000`

Rationale:

- AdamW + `2.5e-4` is a stable baseline for small decoder-only Transformers.
- `batch_size=16` is usually memory-friendly on common GPUs/CPUs while keeping updates reasonably noisy.
- `gradient_clip=1.0` improves stability when random-span batches occasionally produce sharp gradients.
- `max_steps=5000` gives a short, learnable run time for local experimentation.
- Frequent eval/checkpoint intervals make progress visible and failures recoverable.

## About epochs vs steps in this codebase

This training loop is **step-based** (it stops at `max_steps`), not epoch-based.
Because batches are sampled from random contiguous spans in one long token stream, epoch counts are approximate.

\[
\text{approx\_epochs} \approx \frac{\text{max\_steps} \times \text{batch\_size} \times \text{block\_size}}{\text{num\_train\_tokens}}
\]

Read `num_train_tokens` from the training log line:
`Train tokens: ... | Val tokens: ...`.
