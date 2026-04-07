# MicroAlpha Engine

## A C++/Python Microstructure Research Engine

MicroAlpha Engine is a reproducible research project for event-driven limit order book prediction built to showcase:
- clean microstructure reasoning
- C++ feature engineering with Python bindings
- modular quant research pipeline design
- reproducibility, diagnostics, and artifact-driven experimentation

The goal is not to build a production trading system.
The goal is to demonstrate the ability to design and implement a serious research-engineering workflow for high-frequency market data.

---

## TL;DR

This project studies short-horizon midprice direction prediction from LOBSTER order book data using a clean event-driven pipeline.

Main takeaways:
- Pooling multiple tickers improves performance, but not uniformly.
- Predictability is highly heterogeneous across names.
- INTC and MSFT are much easier to predict than AAPL, AMZN, and GOOG under the tested setup.
- Label design matters a lot: dropping zero-return events materially changes the effective learning problem.
- The strongest predictive features are intuitive microstructure signals:
  - best-level queue imbalance
  - microprice deviation
  - short-horizon accumulated OFI
- All features are computed in C++, exposed to Python, and used in a reproducible research pipeline.

A preserved example run is available under:

```docs/example_run/2026-04-07_161228_h500_direction_pooled_5t/```

This reference run contains:
- config snapshot
- metrics
- per-ticker summaries
- diagnostics
- figures
- CSV tables
- run log

Serialized model binaries (`.joblib`) are excluded from the saved run to keep the repository lightweight.

---

## What this repo contains
- Reproducible experiment pipeline for event-driven LOB prediction
- C++ feature engine exposed to Python through pybind11
- Config-driven experiments with per-run artifact generation
- Model comparison between:
  - Logistic Regression
  - HistGradientBoostingClassifier
- Diagnostics and interpretability:
  - feature summaries
  - per-ticker pooled evaluation
  - logistic coefficients
  - permutation importance
- Dockerized execution
- Automated tests
- Saved example run artifacts for easy inspection without rerunning the full pipeline

---

## Research question

Can short-horizon direction be predicted from order book state and recent event flow, and how does predictability change when:
- training on a single ticker vs pooling multiple tickers
- conditioning on price movement vs keeping zero-return events
- comparing linear vs nonlinear models

This repo focuses on research quality and engineering quality, not on execution, alpha monetization, or live deployment.

---

## Main findings

### 1. Pooling helps, but not uniformly

  Pooling 5 tickers improved the pooled headline AUC, but per-ticker analysis showed that:
  - weak names stayed weak or improved only modestly
  - strong names stayed strong
  - pooled performance is partly driven by cross-sectional heterogeneity, not just universal transfer

### 2. Label construction changes the problem

Using `binary_drop_ties` solves:

> direction prediction conditional on price moving

This materially increases apparent predictability for high-tie names like INTC and MSFT.

Using `binary_keep_ties_as_zero` lowers performance, but does not eliminate the effect, which suggests that the phenomenon is real and not purely a filtering artifact.

### 3. Core signal is concentrated in a few sensible features

Across coefficient analysis and permutation importance, the main signal comes from:
- queue_imbalance_best
- microprice_deviation
- ofi_roll_sum_50

The nonlinear tree model extracts additional value from:
- depth imbalance
- short-term volatility
- some longer normalized OFI windows

### 4. Predictability is cross-sectionally heterogeneous

INTC and MSFT are structurally easier than the other names in this sample and at this horizon.
The project carefully narrows down what this does not come from:
- not obvious leakage
- not purely movement prediction
- not purely pooling
- not purely tie filtering

The exact structural cause remains open, but the effect is documented and characterized carefully.

---

## Pipeline at a glance
1. Load LOBSTER message/order book data
2. Compute all features in C++
3. Create labels from forward midprice delta
4. Align features and labels
5. Split each ticker in time
6. Pool train/test sets across tickers if requested
7. Train models
8. Evaluate
9. Save artifacts, figures, tables, diagnostics, and logs

Main entrypoint:

`python -m scripts.run_experiment`

---

## Feature set

All features are computed in C++.

### Core order-book features

- `ofi_best`
- `ofi_best_norm`
- `queue_imbalance_best`
- `depth_imbalance_3`
- `depth_imbalance_5`
- `depth_imbalance_10`
- `spread`
- `microprice_deviation`

### Temporal features

- `ofi_roll_sum_50`
- `ofi_best_norm_roll_sum_10`
- `ofi_best_norm_roll_sum_50`
- `ofi_best_norm_roll_sum_100`
- `midprice_vol_50`
- `event_intensity_1s`

---

## Labels

The project supports multiple binary task formulations.

### Direction conditional on movement

```
task.name = direction
label_mode = binary_drop_ties
```

### Direction with ties kept as class 0

```
task.name = direction 
label_mode = binary_keep_ties_as_zero
```

### Movement prediction

```
task.name = movement 
label_mode = binary
```

### Main results in the saved reference run use:

```
task.name = direction
label_mode = binary_drop_ties
horizon = 500
```

---

## Models

### 1. Logistic Regression

- standardized
- interpretable baseline
- coefficients saved to artifacts

### 2. HistGradientBoostingClassifier

- nonlinear tabular model
- better at exploiting interactions and nonlinear state structure
- permutation importance saved to artifacts

---

## Quickstart

### Local C++ build prerequisites

Building the `_cpp` extension locally requires:
- CMake >= 3.18
- Python >= 3.12
- `pybind11` installed in the active Python environment
- a C++20-capable compiler
  - Windows: Visual Studio Build Tools / MSVC
  - Linux: `g++`
  - macOS: Apple Clang / Xcode command line tools

If local native build setup is inconvenient, use the Docker workflow instead.

### 1. Create environment
```
python -m venv .venv
```

#### Activate it:

Linux/macOS:

```
source .venv/bin/activate
```

Windows:

```
.venv\Scripts\activate
```

### 2. Install

```
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"
```

### 3. Build the C++ extension

Windows PowerShell:

```powershell
.\scripts\build_cpp.ps1
```

Linux/macOS
```bash
cmake -S cpp -B cpp/build \
  -DPYBIND11_FINDPYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --config Release
```

### 4. Put LOBSTER data in place

The config expects CSV files under `data/raw/...` as referenced in:

```
config/experiment.yaml
```

### 5. Run tests

```
pytest -q
```

### 6. Run experiment

```
python -m scripts.run_experiment
```

Artifacts are written under:

`artifacts/<run_id>/`

---

## Docker

Build:

```
docker build -t microalpha .
```

Test:

```
docker run --rm microalpha pytest -q
```

Run:

```
docker run --rm \
  --mount type=bind,source="$(pwd)/data",target=/app/data \
  --mount type=bind,source="$(pwd)/artifacts",target=/app/artifacts \
  microalpha
```
Windows PowerShell

```
docker run --rm `
  --mount "type=bind,source=${PWD}\data,target=/app/data" `
  --mount "type=bind,source=${PWD}\artifacts,target=/app/artifacts" `
  microalpha
```

Notes:

- `data/` is mounted so the container can read local LOBSTER data
- `artifacts/` is mounted so outputs persist on the host
- code and config are copied into the image at build time for reproducibility

---

## Tests

The test suite covers the core invariants of the project:
- config loading smoke test
- LOBSTER loader smoke test
- label alignment correctness
- pooled split / per-ticker test segment consistency
- feature matrix shape / feature-name consistency
- `_cpp` extension import smoke test

Run:
```
pytest -q
```

---

## Example run artifacts

A preserved example run is stored in:

`docs/example_run/2026-04-07_161228_h500_direction_pooled_5t/`

This is included so readers can inspect:
- what a completed run folder looks like
- what files are generated
- how metrics / diagnostics / tables / figures are organized

This saved run excludes model `.joblib` files to keep the repository lightweight.

---

## Reproducibility

The pipeline is designed around:
- config-driven experiments
- per-run artifact folders
- saved config snapshot
- logs
- deterministic train/test splitting by ticker
- explicit pooled test reconstruction for per-ticker pooled evaluation

Each run saves:

- `config.json`
- `metrics.json`
- `split_summary.json`
- `ticker_summaries.json`
- `ticker_feature_diagnostics.json`
- `pooled_ticker_metrics.json`
- permutation importance JSON/CSV
- plots
- run log

---

## Repo structure

```
microalpha-engine/
├── artifacts/                # GENERATED: per-run artifacts
├── config/
│   └── experiment.yaml       # main experiment configuration
├── cpp/                      # C++ feature engine + bindings build
│   ├── include/
│   │   ├── microalpha/
│   │   │   └── features.hpp
│   ├── src/
│   │   ├── bindings.cpp
│   │   └── features.cpp
│   └── CMakeLists.txt
├── data/                     # local input data (not distributed)
├── docs/
│   └── example_run/            # preserved example run artifacts
├── microalpha/               # Python package
│   ├── config.py
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── features.py
│   ├── io.py
│   ├── labels.py
│   ├── models.py
│   ├── pipeline.py
│   └── utils.py
├── scripts/
│   ├── build_cpp.ps1         # script to build C++ module
│   └── run_experiment.py     # main orchestrator
├── tests/                    # automated tests
├── Dockerfile
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Data note

This repository does not include raw LOBSTER market data.  
This project was built using free [LOBSTER](https://data.lobsterdata.com/info/DataSamples.php) sample data for AAPL, AMZN, GOOG, INTC, and MSFT at 10 depth levels.

To run the full pipeline, you must supply your own data in the expected directory structure under `data/raw/....`

Code licensing and data licensing are separate issues.  
This repo distributes the code and example artifacts, not the raw proprietary dataset.

---

## License

This project is licensed under the **BSD 3-Clause License**.  
See the `LICENSE` file for details.

## Contact

**Gautier Petit** 

- [GitHub](https://github.com/gautierpetit): gautierpetit
- [LinkedIn](https://www.linkedin.com/in/gautierpetitch): gautierpetitch