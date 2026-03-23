from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_id(prefix: str = "experiment") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{prefix}"


def make_dataset_prefix(symbols: list[str]) -> str:
    lowered = [symbol.lower() for symbol in symbols]
    if len(lowered) == 1:
        return lowered[0]
    return f"pooled_{len(lowered)}t"


def make_experiment_prefix(
    *,
    horizon: int,
    task_name: str,
    symbols: list[str],
) -> str:
    dataset_prefix = make_dataset_prefix(symbols)
    return f"h{horizon}_{task_name}_{dataset_prefix}"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_artifact_dirs(
    run_id: str, base_dir: str | Path = "artifacts"
) -> dict[str, Path]:
    root = ensure_dir(Path(base_dir) / run_id)
    figures = ensure_dir(root / "figures")
    logs = ensure_dir(root / "logs")
    models = ensure_dir(root / "models")
    tables = ensure_dir(root / "tables")
    return {
        "root": root,
        "figures": figures,
        "logs": logs,
        "models": models,
        "tables": tables,
    }


def setup_logger(
    log_path: str | Path, logger_name: str = "microalpha"
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def stringify_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def save_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    import pandas as pd

    path = Path(path)
    pd.DataFrame(rows).to_csv(path, index=False)
