from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    message_csv: str
    orderbook_csv: str
    levels: int
    price_scale: int


@dataclass(frozen=True)
class LabelConfig:
    horizon: int
    label_mode: str


@dataclass(frozen=True)
class FeatureConfig:
    ofi_window_raw: int
    ofi_norm_windows: list[int]
    vol_window: int
    intensity_window: str


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float


@dataclass(frozen=True)
class LogisticConfig:
    random_state: int


@dataclass(frozen=True)
class HistGBDTConfig:
    random_state: int
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class ModelConfig:
    logistic: LogisticConfig
    hist_gbdt: HistGBDTConfig


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    labels: LabelConfig
    features: FeatureConfig
    split: SplitConfig
    models: ModelConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        labels=LabelConfig(**raw["labels"]),
        features=FeatureConfig(**raw["features"]),
        split=SplitConfig(**raw["split"]),
        models=ModelConfig(
            logistic=LogisticConfig(**raw["models"]["logistic"]),
            hist_gbdt=HistGBDTConfig(**raw["models"]["hist_gbdt"]),
        ),
    )


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset": vars(cfg.dataset),
        "labels": vars(cfg.labels),
        "features": vars(cfg.features),
        "split": vars(cfg.split),
        "models": {
            "logistic": vars(cfg.models.logistic),
            "hist_gbdt": vars(cfg.models.hist_gbdt),
        },
    }