# 7) Serving the Model with an API

## Why an API?

CLI generation is fine for occasional use, but an API is better when:

- You want to send many prompts quickly.
- You want to integrate with a web UI or another app.
- You want one model process loaded in memory and reused.

## What this project exposes

Server code: [api.py](/home/subbu/Code/try-llm/tiny-llm/src/tiny_llm/api.py)

- `GET /health`: basic readiness endpoint.
- `POST /generate`: returns generated text.

Request body:

```json
{
  "prompt": "Once upon a moonlit night,",
  "max_new_tokens": 120,
  "temperature": 0.9,
  "top_k": 40
}
```

## Important serving concept: load once, reuse many times

The model checkpoint and tokenizer are loaded once when the server starts.
Each request only runs inference, which avoids repetitive startup cost.

## Input validation

The API validates generation parameters:

- `prompt` must be non-empty.
- `max_new_tokens` has a safe upper bound.
- `temperature` must be strictly positive.
- `top_k` must be positive when provided.

Invalid input returns HTTP 400 with clear details.

