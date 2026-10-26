from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset_name: str
    split: str
    text_field: str
    max_samples: int
    val_ratio: float
    vocab_size: int
    min_frequency: int
    tokenizer_path: str


@dataclass(slots=True)
class ModelConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    max_steps: int
    eval_interval: int
    eval_iters: int
    gradient_clip: float
    checkpoint_dir: str
    checkpoint_every: int
    log_every: int
    preview_tokens: int


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        return cls(
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(
                **{
                    **raw["training"],
                    "betas": tuple(raw["training"]["betas"]),
                }
            ),
        )

