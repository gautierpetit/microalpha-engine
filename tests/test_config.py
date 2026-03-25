from pathlib import Path

from microalpha.config import config_to_dict, load_experiment_config


def test_load_experiment_config_smoke(tmp_path: Path) -> None:
    cfg_path = tmp_path / "experiment.yaml"
    cfg_path.write_text(
        """
dataset:
  levels: 10
  price_scale: 10000
  tickers:
    - symbol: TEST
      message_csv: data/raw/test_message.csv
      orderbook_csv: data/raw/test_orderbook.csv

task:
  name: direction

labels:
  horizon: 500
  label_mode: binary_drop_ties

features:
  ofi_window_raw: 50
  ofi_norm_windows: [10, 50, 100]
  vol_window: 50
  intensity_window: "1s"

split:
  train_fraction: 0.7

models:
  logistic:
    random_state: 42
  hist_gbdt:
    random_state: 42
    learning_rate: 0.05
    max_iter: 200
    max_leaf_nodes: 31
    min_samples_leaf: 50
    l2_regularization: 0.0
""",
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_path)

    assert cfg.dataset.levels == 10
    assert cfg.dataset.price_scale == 10000
    assert len(cfg.dataset.tickers) == 1
    assert cfg.dataset.tickers[0].symbol == "TEST"
    assert isinstance(cfg.dataset.tickers[0].message_csv, Path)
    assert cfg.task.name == "direction"
    assert cfg.labels.label_mode == "binary_drop_ties"
    assert cfg.features.ofi_norm_windows == [10, 50, 100]
    assert cfg.split.train_fraction == 0.7


def test_config_to_dict_serializes_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "experiment.yaml"
    cfg_path.write_text(
        """
dataset:
  levels: 10
  price_scale: 10000
  tickers:
    - symbol: TEST
      message_csv: data/raw/test_message.csv
      orderbook_csv: data/raw/test_orderbook.csv

task:
  name: direction

labels:
  horizon: 500
  label_mode: binary_drop_ties

features:
  ofi_window_raw: 50
  ofi_norm_windows: [10, 50, 100]
  vol_window: 50
  intensity_window: "1s"

split:
  train_fraction: 0.7

models:
  logistic:
    random_state: 42
  hist_gbdt:
    random_state: 42
    learning_rate: 0.05
    max_iter: 200
    max_leaf_nodes: 31
    min_samples_leaf: 50
    l2_regularization: 0.0
""",
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_path)
    d = config_to_dict(cfg)

    assert isinstance(d["dataset"]["tickers"][0]["message_csv"], str)
    assert isinstance(d["dataset"]["tickers"][0]["orderbook_csv"], str)
