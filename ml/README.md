# sleepguard-ml

ML package for SleepGuard. Owner: Inaaya (+ Model member).
Covers interface I2 (SGR format) and I4 (`analyze()`), dataset converters, the M0 baseline and evaluation metrics.

## Setup (Windows, VS Code terminal)

```powershell
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[convert,dev]"
pytest
```

All 13 tests should pass.

## What is here

| File | What |
| --- | --- |
| `sleepguard_ml/schemas.py` | `Recording` (SGR v1), `AnalysisResult`. Shared with the backend |
| `sleepguard_ml/io.py` | SGR JSON / Parquet read and write |
| `sleepguard_ml/preprocess.py` | Resample to 1 Hz, clean out-of-range values |
| `sleepguard_ml/baseline.py` | M0: ODI rule + linear AHI calibration |
| `sleepguard_ml/api.py` | `load_model()`, `analyze()` for the backend |
| `sleepguard_ml/metrics.py` | Event F1, AHI agreement (MAE, ICC, Bland-Altman), severity kappa, screening, bootstrap CI |
| `sleepguard_ml/synthetic.py` | Synthetic nights with labels |
| `sleepguard_ml/converters/` | UCD (PhysioNet) and SHHS / MESA (NSRR) to SGR |
| `scripts/` | Command line tools (below) |

## Commands

```powershell
# synthetic data for the app / backend / dashboard team
python scripts/make_synthetic.py --out ../data/synthetic --n 40 --examples ../docs/interfaces/examples

# M0 on any labelled folder (5-fold calibration)
python scripts/eval_baseline.py --data ../data/synthetic --cv 5
```

**Synthetic results mean nothing.** The generator makes clean dips, so M0 looks almost perfect on it.
Use synthetic data only to test code. Real numbers come from UCD, SHHS and MESA.

## Real data, step by step

### 1. UCD (open, start now)

1. Download the St. Vincent's University Hospital / UCD Sleep Apnea Database from PhysioNet into `data/raw/ucd/`.
2. Save `SubjectDetails` as CSV in the same folder.
3. Check the events parser on one file first:
   ```python
   from sleepguard_ml.converters.ucd import debug_respevt
   debug_respevt("../data/raw/ucd/ucddb002_respevt.txt")
   ```
   Compare the parsed events with the raw lines. Fix `parse_respevt` if columns do not match.
4. Convert, then evaluate:
   ```powershell
   python scripts/convert_ucd.py --src ../data/raw/ucd --out ../data/sgr/ucd --ahi-csv ../data/raw/ucd/SubjectDetails.csv
   python scripts/eval_baseline.py --data ../data/sgr/ucd --cv 5
   ```
   If `pyedflib` will not open a `.rec` file, copy it with an `.edf` extension.

### 2. SHHS / MESA (after NSRR approval, Inaaya only)

1. Download with the NSRR tools into `data/raw/shhs/` on the secure laptop.
2. Pick ONE AHI variable from the NSRR data dictionary. Write it in the model card.
3. Test on 5 files first: `python scripts/convert_nsrr.py --source shhs ... --limit 5`
4. Then convert all.

## NSRR rules (do not break these)

- SHHS / MESA files and converted Parquet files stay on Inaaya's secure laptop. Never in git, Drive or Colab.
- Share results (tables, metric plots) only. Never a participant's signal plot.
- Acknowledge `NSRR R24 HL114473` and cite SHHS / MESA in every paper or slide deck.

## Next (plan Section 11.4)

- Week 2: UCD converted, M0 results on UCD
- Week 3: M1 features + XGBoost
- Week 4: M2 CNN-BiLSTM, ONNX export, `load_model("*.onnx")`
