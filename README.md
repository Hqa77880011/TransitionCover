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

Validation uses Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, PyTorch 2.7.1+cu118 (CPU), scikit-learn 1.6.1 and matplotlib 3.10.0. `python tests/functional_check.py` executes NPZ loading, training/development distinction-map fitting, grouped preparation, a one-epoch train/evaluate/plot cycle, finite loss/backward and key GRU/head updates, save/load equality (rtol 1e-5, atol 1e-6), and a hand-calculated two-transition case with delays 1/1 and deadline success 1. Seed 13; OMP/MKL/torch threads 1. No full training or original dataset replay was run.

## Paper-to-code correspondence

| Module / metric | Entry | Setting / command |
|---|---|---|
| Transition graph | `Model.prepare` | training state transitions only |
| Group distinction map | `Model.prepare` | logistic discriminators train-only; development balanced accuracy >=0.78 |
| Group-union cover and wake cost | `Model.forward` | singleton and configured paired groups; overlapping endpoints charged once |
| Pair ages, full-probe reservation | `Model.forward` | completed available groups reset ages; explicit per-window command allowance |
| Causal recognition | `Model.forward` | two GRUCell layers; value/mask/elapsed inputs, width 64 |
| Event recall / deadline success / false notifications per hour | `method.measure` | one-to-one matching, missed events remain in denominator |
| State macro-F1 / unknown fraction / delay p95 | `method.measure` | missing ground-truth states excluded; controller unknown remains a failed prediction |
| Endpoint energy proxy / wakes | `method.measure` | 0.15+0.85*attempted fraction+0.0003*wakes per endpoint-hour |

`available[N,T,S]` optionally supplies replay completion availability. Failed attempts consume command allowance but never enter recognition or reset pair ages. `attempted_readout` and completed `readout` are exported separately. The age guard emits unknown after `probe_interval+confirmation` unchecked intervals for required graph edges. `interval_s`, `deadline_steps`, `probe_interval`, `window` and `confirmation` define time units; set them for the native recording grid. The demo uses one-second intervals and three classes; real source-specific six-class dictionaries and 12 endpoints must be exported without merging distinct source meanings. `paired_groups` is a list of zero-based endpoint-index lists. The default uses adjacent pairs.

The logistic map uses causal rolling means (`feature_window=5`), a compact implementation choice for the paper's causal group features. The replay model supports availability and completion at the current grid tick; physical settling/transfer latency, minimum on-holds, quarantining/re-entry handshakes and burst-reserve accounting are not modeled as a hardware simulator. Reported latency is host software time and the endpoint energy is an assumed proxy. The original OPPORTUNITY/CASAS session manifest, raw replay artifacts and measured gateway costs were not present in this repository. The supplied generator is a functional fixture, separate from original experiment evidence. Run sources separately and average their source-level metrics equally when reproducing the paper aggregation.
