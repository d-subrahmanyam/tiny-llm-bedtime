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

