from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml


TaskName = Literal["direction", "movement"]


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    message_csv: Path
    orderbook_csv: Path


@dataclass(frozen=True)
class DatasetConfig:
    tickers: list[TickerConfig]
    levels: int
    price_scale: int


@dataclass(frozen=True)
class TaskConfig:
    name: TaskName


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
    task: TaskConfig
    labels: LabelConfig
    features: FeatureConfig
    split: SplitConfig
    models: ModelConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw_dataset = raw["dataset"]
    raw_tickers = raw_dataset.get("tickers")
    if not raw_tickers:
        raise ValueError("dataset.tickers must be provided and non-empty")

    return ExperimentConfig(
        dataset=DatasetConfig(
            tickers=[TickerConfig(**ticker) for ticker in raw_tickers],
            levels=raw_dataset["levels"],
            price_scale=raw_dataset["price_scale"],
        ),
        task=TaskConfig(**raw["task"]),
        labels=LabelConfig(**raw["labels"]),
        features=FeatureConfig(**raw["features"]),
        split=SplitConfig(**raw["split"]),
        models=ModelConfig(
            logistic=LogisticConfig(**raw["models"]["logistic"]),
            hist_gbdt=HistGBDTConfig(**raw["models"]["hist_gbdt"]),
        ),
    )


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)
