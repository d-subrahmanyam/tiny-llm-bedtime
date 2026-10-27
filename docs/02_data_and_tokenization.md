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

## Config chosen in this project (and why)

From [configs/default.yaml](/home/subbu/Code/try-llm/tiny-llm/configs/default.yaml):

- `vocab_size: 4096`
- `min_frequency: 2`
- tokenizer implementation: byte-level BPE (`ByteLevel` pre-tokenizer + decoder)
- special tokens: `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>`

Rationale:

- `4096` keeps embedding/output layers compact for a tiny model while still giving enough subword coverage for bedtime vocabulary.
- `min_frequency=2` prevents single-occurrence artifacts from occupying vocabulary slots.
- Byte-level preprocessing avoids hard failures on rare characters and punctuation.
- Explicit BOS/EOS markers make sequence boundaries visible to the model.
