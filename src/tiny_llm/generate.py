from __future__ import annotations

import argparse
from pathlib import Path

from tiny_llm.inference import generate_text, load_inference_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained tiny GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    model, tokenizer, device = load_inference_stack(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
