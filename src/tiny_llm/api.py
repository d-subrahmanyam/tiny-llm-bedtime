from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tiny_llm.inference import generate_text, load_inference_stack


class GenerateRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, description="Prompt text for the model.")]
    max_new_tokens: Annotated[int, Field(ge=1, le=512)] = 120
    temperature: Annotated[float, Field(gt=0.0, le=2.5)] = 0.9
    top_k: Annotated[int | None, Field(ge=1, le=200)] = 40


class GenerateResponse(BaseModel):
    text: str


def create_app(config_path: Path, checkpoint_path: Path) -> FastAPI:
    app = FastAPI(title="Tiny LLM API", version="0.1.0")
    # Load once on startup; each request only runs generation.
    model, tokenizer, device = load_inference_stack(config_path=config_path, checkpoint_path=checkpoint_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            text = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )
            return GenerateResponse(text=text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Tiny LLM via FastAPI.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(config_path=args.config, checkpoint_path=args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
