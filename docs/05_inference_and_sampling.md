# 5) Inference and Sampling

Inference code is in [generate.py](/home/subbu/Code/try-llm/tiny-llm/src/tiny_llm/generate.py).

## Autoregressive generation

1. Encode prompt into token IDs.
2. Feed into model.
3. Take logits for the last position.
4. Convert logits to probabilities.
5. Sample one token.
6. Append token and repeat.

## Temperature

- `< 1.0`: more conservative, repetitive.
- `= 1.0`: neutral.
- `> 1.0`: more diverse, riskier.

## Top-k sampling

Keep only top `k` probable tokens; set others to probability 0.
This often improves text quality for small models.

