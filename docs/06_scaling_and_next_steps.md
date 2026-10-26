# 6) Scaling and Next Steps

## Better data

- Increase dataset size.
- Clean noisy rows.
- Add metadata-aware prompts (optional).

## Better model

- Increase depth (`n_layer`) and width (`n_embd`).
- Increase context (`block_size`).
- Keep attention heads dividing embedding size exactly.

## Better training

- Longer schedule.
- Learning-rate warmup + decay.
- Mixed precision (GPU).
- Gradient accumulation for bigger effective batch.

## Better evaluation

- Track perplexity (`exp(loss)`).
- Keep fixed validation split.
- Store sample generations at each eval step.

## Safety and practical caveats

- Small models can still produce flawed content.
- Never rely on generated text for factual correctness.
- Treat this project as an educational base for further experimentation.

