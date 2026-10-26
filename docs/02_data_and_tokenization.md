# 2) Data and Tokenization

## Dataset from Hugging Face

The code uses `datasets.load_dataset(dataset_name, split=...)`.

Key parameters in [default.yaml](/home/subbu/Code/try-llm/tiny-llm/configs/default.yaml):

- `dataset_name`: Hugging Face dataset ID
- `text_field`: column containing raw story text
- `max_samples`: training subset size for fast iteration

## Why tokenization matters

Neural networks do not consume raw strings. They consume integers.
Tokenizer maps text chunks to IDs.

Example:

`"Once upon a time"` -> `[421, 98, 15, 713]`

## BPE tokenizer (used here)

Byte-Pair Encoding starts from small units and merges frequent pairs. Benefits:

- Handles unknown words by composing subwords.
- Smaller vocabulary than word-level tokenizers.
- Better robustness for names/spelling variations.

## Special tokens

- `<|bos|>`: begin sequence
- `<|eos|>`: end sequence
- `<|unk|>`: unknown token
- `<|pad|>`: padding token (not central in this simple training loop)

`<|eos|>` is appended between stories so the model learns boundaries.

