# SleepGuard

Low-cost AI screening for obstructive sleep apnea using overnight pulse oximetry.
Student project for the AIconomy International AI Challenge 2026/27 (University of Peradeniya).

| Folder | What | Status |
| --- | --- | --- |
| `ml/` | `sleepguard_ml` package: SGR format, converters, baseline, metrics | Started |
| `firmware/` | ESP32 firmware (PlatformIO) | Planned |
| `app/` | Flutter phone app | Planned |
| `backend/` | FastAPI backend | Planned |
| `dashboard/` | React clinician dashboard | Planned |

## Data rule

No NSRR (SHHS, MESA) files, patient data or split ID lists in this repo. Ever.
The `data/` folder is git-ignored. See `ml/README.md`.

This is a screening research prototype, not a medical device.
