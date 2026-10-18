# TransitionCover

Implementation of **TransitionCover: Budgeted Sensor Observability for Responsive Smart-Home Activity Services**.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`x[N,T,S]` contains synchronous replay features, `y[N]` final activity label, and `state[N,T]` the timeline annotation used only for training/evaluation. Keep subject/home/session/day groups intact. For event channels preserve release semantics before exporting; absence of an event is not a fabricated observation. OPPORTUNITY/CASAS can be exported with a fixed annotated time grid.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent source units; related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

`demo` generates synthetic inputs for this tensor interface; application-specific results require acquired data.

```bash
transitioncover demo --output data/example.npz --count 96 --config config.json --seed 13
transitioncover prepare --input data/example.npz --output data/prepared
```

## Training and inference

```bash
transitioncover train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

## Evaluation

```bash
transitioncover evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
transitioncover plot --input runs --output runs/figures
```

## Ablations

Supported controls: `baseline`, `no_probe`, `no_age`, `no_wake`. Keep the input groups, seed and budget fixed.

```bash
transitioncover train --input data/prepared --config config.json --output runs/no_probe --variant no_probe --epochs 100 --batch-size 16 --seed 13
```

## Outputs

`data/prepared/` contains train/validation/calibration/test NPZs. Training writes `model.pt`, `history.json` and `training.json` with run arguments. Evaluation writes `predictions.npz` and `metrics.json`; plotting reads saved metric files.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
