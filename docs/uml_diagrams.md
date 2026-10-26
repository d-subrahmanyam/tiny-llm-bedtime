# UML Diagrams for the Tiny LLM Project

These diagrams use Mermaid syntax and can be rendered by many Markdown viewers.

## 1) Component Diagram (Pipeline Overview)

```mermaid
flowchart LR
    A[Hugging Face Dataset] --> B[Data Loader]
    B --> C[Tokenizer Trainer / Loader]
    C --> D[Token IDs]
    D --> E[Batch Sampler]
    E --> F[TinyGPT Model]
    F --> G[Cross-Entropy Loss]
    G --> H[AdamW Optimizer]
    H --> F
    F --> I[Checkpoint Saver]
    F --> J[Text Generator]
```

## 2) Class Diagram (Core Model Classes)

```mermaid
classDiagram
    class TinyGPT {
      +int block_size
      +forward(idx, targets) (logits, loss)
      +generate(idx, max_new_tokens, temperature, top_k) tokens
    }
    class Block {
      +forward(x) x
    }
    class CausalSelfAttention {
      +forward(x) x
    }
    class MLP {
      +forward(x) x
    }
    TinyGPT --> Block : contains N
    Block --> CausalSelfAttention : uses
    Block --> MLP : uses
```

## 3) Sequence Diagram (Training Step)

```mermaid
sequenceDiagram
    participant Trainer
    participant Data as DataModule
    participant Model as TinyGPT
    participant Opt as AdamW

    Trainer->>Data: get_batch(train_tokens)
    Data-->>Trainer: x, y
    Trainer->>Model: forward(x, y)
    Model-->>Trainer: logits, loss
    Trainer->>Opt: zero_grad()
    Trainer->>Trainer: loss.backward()
    Trainer->>Trainer: clip_grad_norm_()
    Trainer->>Opt: step()
```

## 4) Activity Diagram (Generation)

```mermaid
flowchart TD
    A[Start with prompt tokens] --> B[Crop to block_size context]
    B --> C[Model forward pass]
    C --> D[Take last-step logits]
    D --> E[Apply temperature and top-k]
    E --> F[Sample next token]
    F --> G[Append token to sequence]
    G --> H{Reached max_new_tokens?}
    H -- No --> B
    H -- Yes --> I[Decode tokens to text]
```

## 5) Sequence Diagram (API Inference Request)

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Server
    participant Model as TinyGPT
    participant Tok as Tokenizer

    Client->>API: POST /generate {prompt, max_new_tokens, temperature, top_k}
    API->>Tok: encode(prompt)
    Tok-->>API: token IDs
    API->>Model: generate(token IDs, sampling args)
    Model-->>API: output token IDs
    API->>Tok: decode(output IDs)
    Tok-->>API: generated text
    API-->>Client: 200 OK {text}
```
