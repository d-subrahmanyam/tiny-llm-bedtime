from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from tiny_llm.config import ExperimentConfig
from tiny_llm.model import TinyGPT
from tiny_llm.train import resolve_device


def load_inference_stack(
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[TinyGPT, Tokenizer, torch.device]:
    cfg = ExperimentConfig.from_yaml(config_path)
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)
    if tokenizer.decoder is None:
        # Backward compatibility for tokenizers saved without a decoder.
        tokenizer.decoder = ByteLevelDecoder()
    vocab_size = tokenizer.get_vocab_size()
    device = resolve_device(cfg.device)

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    ).to(device)

    payload = torch.load(checkpoint_path, map_location=device)
    # Checkpoint stores model_state_dict alongside optimizer/training metadata.
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_text(
    model: TinyGPT,
    tokenizer: Tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    encoded = tokenizer.encode(prompt)
    if not encoded.ids:
        raise ValueError("Prompt produced no tokens. Use a non-empty prompt.")

    start = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    output = model.generate(
        idx=start,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(output[0].tolist())
