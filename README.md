# ULD-IG

Code for **Scaling Laws under Limited Budgets: Predicting LLM Fine-Tuning Loss**.

## Installation

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python run.py --formula ULD-IG --fit-method soft --dataset SFT-15 --budget 9600
python run.py --help
```

| Formula | Fitting methods |
|---|---|
| ULD-IG | `default`, `hard`, `soft` |
| Rectified | `default`, `soft` |
| SLD | `default`, `soft` |
| Power | `default` |
| Vanilla | `default` |

`default` uses unanchored fitting; `hard` and `soft` apply hard and soft initial-loss anchoring. Omit `--budget` to run all configured budgets.

## Data and output

The [data/](data/) directory contains:

| Dataset | Curves | Source |
|---|---:|---|
| Lin24Rect | 90 | [Rectified Scaling Law](https://github.com/linhaowei1/Fine-tuning-Scaling-Law) |
| SLDBench | 42 | SFT subset from [SLD](https://github.com/linhaowei1/SLD) |
| SFT-15 | 15 | This work |

SLDBench has no measured initial loss and uses `default` fitting.

Settings are in [configs/default.yaml](configs/default.yaml). Fitting results are saved as JSON under `results/`.
