from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)

        q = q.view(bsz, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, time, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, time, channels)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"Sequence length {time} exceeds block_size {self.block_size}.")

        pos = torch.arange(0, time, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

