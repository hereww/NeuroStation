# SSVEP Benchmark / Wang2016

## Result

Complete public mirror downloaded and verified: **35 subject MAT files plus `Readme.txt`, `Sub_info.txt`, and `64-channels.loc`**. The Zenodo mirror exposed 38 files totaling about 3.71 GB; all downloaded files match the source-reported byte sizes. Every file has a SHA256 entry in `DATASET_MANIFEST.json`.

## Task-relevant facts

- SSVEP 40-target JFPM benchmark; 35 subjects; 6 blocks × 40 trials.
- Source acquisition: 1000 Hz, 64 EEG channels; supplied epochs are downsampled to 250 Hz and shaped as `[64, 1500, 40, 6]` in the published description.
- Stimulus frequencies: 8.0–15.8 Hz in 0.2 Hz steps; target order/frequency-phase mapping is described in the source README/paper. Labels are represented by target/block tensor order rather than a separate stim channel in these MAT files.
- License: CC BY 4.0 for the Zenodo record.

## Caveat

The Zenodo mirror did not include the paper's `Freq_Phase.mat`; the frequency/phase mapping should therefore be treated as source metadata from the paper/README, not inferred from an absent local file. `CB1`/`CB2` location ambiguity also remains relevant for spatial analyses.

## Provenance

Source record: https://zenodo.org/records/14865172
DOI: 10.1109/TNSRE.2016.2627556
