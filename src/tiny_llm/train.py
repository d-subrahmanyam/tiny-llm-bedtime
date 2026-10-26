from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import trange

from tiny_llm.config import ExperimentConfig
from tiny_llm.data import encode_corpus, get_batch, load_texts, split_train_val, train_or_load_tokenizer
from tiny_llm.model import TinyGPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_tokens), ("val", val_tokens)):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Loss unexpectedly None during evaluation.")
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def decode_preview(
    model: TinyGPT,
    tokenizer: Tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    bos_id = tokenizer.token_to_id("<|bos|>")
    if bos_id is None:
        raise ValueError("Tokenizer missing <|bos|> token.")

    prompt = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    output = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.9, top_k=40)
    ids = output[0].tolist()
    return tokenizer.decode(ids)


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    config_path: Path,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config_path": str(config_path),
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny GPT model on bedtime stories.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    texts = load_texts(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.split,
        text_field=cfg.data.text_field,
        max_samples=cfg.data.max_samples,
    )
    print(f"Loaded {len(texts)} texts from {cfg.data.dataset_name}.")

    tokenizer = train_or_load_tokenizer(
        texts=texts,
        tokenizer_path=cfg.data.tokenizer_path,
        vocab_size=cfg.data.vocab_size,
        min_frequency=cfg.data.min_frequency,
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer ready with vocab size: {vocab_size}")

    tokens = encode_corpus(tokenizer, texts)
    train_tokens, val_tokens = split_train_val(tokens, cfg.data.val_ratio)
    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
    )

    print("Starting training...")
    for step in trange(1, cfg.training.max_steps + 1):
        xb, yb = get_batch(train_tokens, cfg.training.batch_size, cfg.model.block_size, device)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss unexpectedly None during training.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        optimizer.step()

        if step % cfg.training.log_every == 0:
            print(f"step={step} train_loss={loss.item():.4f}")

        if step % cfg.training.eval_interval == 0:
            metrics = estimate_loss(
                model=model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                batch_size=cfg.training.batch_size,
                block_size=cfg.model.block_size,
                eval_iters=cfg.training.eval_iters,
                device=device,
            )
            print(
                f"[eval] step={step} train={metrics['train']:.4f} val={metrics['val']:.4f}"
            )
            sample = decode_preview(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=cfg.training.preview_tokens,
            )
            print("-" * 80)
            print(sample)
            print("-" * 80)

        if step % cfg.training.checkpoint_every == 0 or step == cfg.training.max_steps:
            saved = save_checkpoint(
                ckpt_dir=Path(cfg.training.checkpoint_dir),
                step=step,
                model=model,
                optimizer=optimizer,
                config_path=args.config,
            )
            print(f"Saved checkpoint: {saved}")

    print("Training complete.")


if __name__ == "__main__":
    main()

