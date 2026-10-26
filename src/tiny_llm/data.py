from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def load_texts(
    dataset_name: str,
    split: str,
    text_field: str,
    max_samples: int,
) -> list[str]:
    dataset = load_dataset(dataset_name, split=split)
    total = min(max_samples, len(dataset))
    texts: list[str] = []
    for row in dataset.select(range(total)):
        value = row.get(text_field)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    if not texts:
        raise ValueError(
            f"No valid text rows found in dataset={dataset_name!r} field={text_field!r}."
        )
    return texts


def train_or_load_tokenizer(
    texts: Iterable[str],
    tokenizer_path: str | Path,
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Keep pre-tokenizer and decoder explicit so legacy tokenizer files
        # still decode clean text (without raw byte-level space markers).
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.save(str(tokenizer_path))
        return tokenizer

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def encode_corpus(tokenizer: Tokenizer, texts: Iterable[str]) -> torch.Tensor:
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing required special token <|eos|>.")

    token_buffer: list[int] = []
    for text in texts:
        enc = tokenizer.encode(text)
        token_buffer.extend(enc.ids)
        token_buffer.append(eos_id)

    if len(token_buffer) < 2:
        raise ValueError("Encoded token buffer is too short to train.")

    return torch.tensor(token_buffer, dtype=torch.long)


def split_train_val(tokens: torch.Tensor, val_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < val_ratio < 0.5):
        raise ValueError("val_ratio must be in range (0.0, 0.5).")

    cut = int(len(tokens) * (1.0 - val_ratio))
    train_tokens = tokens[:cut]
    val_tokens = tokens[cut:]
    if len(train_tokens) < 100 or len(val_tokens) < 100:
        raise ValueError(
            "Not enough tokens for train/val split. Increase max_samples or lower val_ratio."
        )
    return train_tokens, val_tokens


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - block_size - 1
    if max_start <= 0:
        raise ValueError("Token stream smaller than block_size. Reduce block_size.")

    # Sample random contiguous spans from the long token stream.
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
