# 3) Transformer Architecture (Causal GPT-style)

Implementation is in [model.py](/home/subbu/Code/try-llm/tiny-llm/src/tiny_llm/model.py).

## Main components

1. **Token Embedding**: maps token IDs -> vectors.
2. **Positional Embedding**: adds order information.
3. **N Transformer Blocks**:
   - LayerNorm
   - Causal self-attention
   - Residual connection
   - MLP feed-forward
4. **Final LayerNorm + Linear head** -> logits over vocabulary.

## Causal mask

In autoregressive modeling, token at position `t` must not see future token `t+1`.
Lower-triangular mask enforces this.

## Attention formula

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
\]

Where `M` is the causal mask (future positions get `-inf`).

## Why residuals + normalization?

- Residual connections help gradient flow.
- LayerNorm stabilizes activations and training.

## Config chosen in this project (and why)

From [configs/default.yaml](/home/subbu/Code/try-llm/tiny-llm/configs/default.yaml):

- `block_size: 192`
- `n_layer: 6`
- `n_head: 6`
- `n_embd: 384`
- `dropout: 0.15`
- `bias: false`

Rationale:

- `block_size=192` keeps attention cost practical while giving enough context for short story structure.
- `6 x 384` (with 6 heads) is a compact but meaningful capacity point for educational training.
- `dropout=0.15` adds regularization for a relatively small corpus budget.
- `bias=false` follows common GPT simplifications and slightly reduces parameter count.
