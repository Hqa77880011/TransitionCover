# TransitionCover

**TransitionCover: Budgeted Sensor Observability for Responsive Smart-Home Activity Services**.

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

## Modules and metrics

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

The distinction map uses causal rolling means (`feature_window=5`). Instant-completion and delayed-completion paths share the recognizer interface. Reported latency measures host software time; endpoint energy uses the configured proxy. Run each source separately and average source-level metrics equally.

## Completion-time replay and source protocol

Use `physical_config.json` for the 12-endpoint/six-class contract: 1s grid, 20s full-probe spacing, 300s accounting windows, minimum on-time 3s, settling 3s, acquisition 1s, transfer 0.5s, 10% burst reserve. A wake reserves its entire minimum observation/transfer commitment; future full probes are reserved before optional reads. Completions retain fractional timestamps and are consumed at the next causal grid tick. Readout commands precede access to that tick's values. Failed acquisitions/transfers never reset distinction ages; communication timeout quarantines endpoints, and a successful fresh retry is required for re-entry. Missing reservations are reported; budget is never silently exceeded. All event channels still require correct release semantics in the supplied recording.

```bash
python -m transitioncover.cli train --input data/prepared --config physical_config.json --output runs/physical --epochs 100 --seed 13
python -m transitioncover.physical --input data/prepared/test.npz --model runs/physical/model.pt --output runs/physical/evaluation
```

`physical=true` routes training, validation and inference through the same delayed-completion scheduler and differentiable value/mask/elapsed recognizer. `no_reserve` is a timing control. `available[N,T,S]` describes eventual replay completion success and is hidden from selection until acknowledgement/timeout. `completion_ledger.json` records wakes, quarantines, probe admission, burst release and remaining allowance. With `source[N]`, metrics are computed within source and receive equal source weights. This is replay of declared physical timing, not a measured radio prototype or a calibrated battery-life estimate. Endpoint attempt charging is conservative and distinct from successful-completion fraction.

Physical-path checks: `python tests/physical_check.py`.
