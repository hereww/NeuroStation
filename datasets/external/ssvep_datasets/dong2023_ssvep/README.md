# Dong2023 SSVEP EEG Dataset

## Result

Complete public Zenodo mirror downloaded and verified: **59 subject MAT files**. Each file matches the source-reported byte size and has a SHA256 entry in `DATASET_MANIFEST.json`.

## Task-relevant facts

- 40-target SSVEP JFPM task; 59 healthy adolescent subjects; 4 blocks × 40 trials.
- Stored data: 250 Hz, 8 EEG channels (`POz`, `PO3`, `PO4`, `PO7`, `PO8`, `Oz`, `O1`, `O2`); each subject contains `eegdata` with documented shape `[8, 1250, 40, 4]`.
- Stimulus frequencies: 8.0–15.8 Hz in 0.2 Hz steps. Labels are represented by target/block tensor order; no separate event channel is documented in the stored MAT structure.
- License: CC BY-NC 4.0 for the Zenodo mirror.

## Caveat

The dataset is a rehosted/mirrored copy of the authors' public data. Check the non-commercial restriction before using it in a commercial or clinical product.

## Provenance

Source record: https://zenodo.org/records/18847318
DOI: 10.26599/BSA.2023.9050020
