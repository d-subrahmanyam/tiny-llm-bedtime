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

