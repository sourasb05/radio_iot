# IoT Intrusion Detection — Per-Domain LSTM Baseline

This project trains independent LSTM classifiers for each network domain in an RPL-based IoT dataset. Each domain represents a unique combination of attack type, network size, and behavioral variant. The goal is to establish per-domain baselines before applying continual learning methods.

---

## Project Structure

```
code/
├── attack_data/                   # Raw data (outside src/)
│   ├── blackhole/
│   │   ├── domain10/
│   │   │   ├── features_energy_60_sec.csv
│   │   │   └── ...
│   │   └── ...
│   ├── dis_flooding/
│   ├── local_repair/
│   ├── worst_parent/
│   └── (failing_node excluded)
├── domain_details.xlsx            # Maps domain folders to AttackType_Node_Version labels
└── src/
    ├── main.py                    # Entry point
    ├── train.py                   # Training loop, evaluation, model saving
    ├── models.py                  # LSTM model definition
    └── utils.py                   # Data loading, preprocessing, argument parsing
```

---

## Dataset

- **48 domains** across 4 attack types: `blackhole`, `dis_flooding`, `local_repair`, `worst_parent`
- Each domain has **20 CSV files** (independent network simulations)
  - 16 files → training, 4 files → test (fixed split via `random.seed(42)`)
- Each CSV has 861 time steps × **18 features** (RPL control message stats for 2 nodes)
- **Features:** `rank, disr, diss, dior, dios, diar, tots, tx, rx` × 2 nodes (`.1` suffix for second node)
- **Excluded:** `cpu`, `cpu.1` (dropped globally)
- **Label:** `0` = benign, `1` = attack

Domain names follow the format `AttackType_NodeCount_Version` (e.g. `dis_flooding_15_gc`), derived from `domain_details.xlsx`.

---

## Feature Experiments

Controlled via `--exp_no` from the command line:

| Exp | Features used |
|-----|--------------|
| 1 | All 18 features |
| 2 | `tx, rx, tx.1, rx.1` (4 features) |
| 3 | All except `rx, rx.1` (16 features) |
| 4 | All except `tx, tx.1` (16 features) |

LSTM input size is computed automatically as `num_features × window_size`.

---

## Requirements

```
python >= 3.9
torch
numpy
pandas
scikit-learn
tqdm
scipy
openpyxl
```

---

## Usage

```bash
cd src/

# Train all 48 domains with all features
python main.py --exp_no 1

# Train a single domain
python main.py --exp_no 1 --domain dis_flooding_15_gc

# Train with tx/rx features only
python main.py --exp_no 2 --domain all

# Common options
python main.py --exp_no 1 \
    --epochs 50 \
    --hidden_size 64 \
    --window_size 10 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --patience 5
```

---

## Outputs

**Saved models:** `src/saved_models/exp{N}/{domain_key}.pt`
- Best model (lowest validation loss) is saved per domain via early stopping.

**Results:** `src/results/exp_features_{N}/{domain_key}/metrics.json`

Each `metrics.json` contains:
```json
{
    "accuracy": 0.9712,
    "f1": 0.9698,
    "precision": 0.9801,
    "recall": 0.9597,
    "auc": 0.9934,
    "confusion_matrix": [[1688, 23], [45, 1648]]
}
```

**Logs:** `src/logs/exp{N}_log_{timestamp}.log`

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_no` | `1` | Feature experiment (1–4) |
| `--domain` | `all` | Domain to train (`all` or specific name) |
| `--epochs` | `3` | Max training epochs |
| `--window_size` | `10` | Sliding window size |
| `--batch_size` | `128` | Batch size |
| `--hidden_size` | `10` | LSTM hidden units |
| `--num_layers` | `1` | LSTM layers |
| `--learning_rate` | `0.001` | Adam learning rate |
| `--patience` | `2` | Early stopping patience |
