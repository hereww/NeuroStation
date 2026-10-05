# Nakanishi2015 12-class SSVEP

## Result

All 10 public subject MAT files from the GitHub data directory were downloaded and verified against the GitHub API file sizes. Each file has a SHA256 entry in `DATASET_MANIFEST.json`.

## Task-relevant facts

- 12-class joint frequency-phase modulated SSVEP; 10 subjects; 15 trials per class.
- Stored tensor `eeg` is documented as `[12 targets, 8 channels, 1114 samples, 15 trials]` at 256 Hz. Stimulus onset is sample 39 (approximately 0.15 s pre-stimulus padding).
- Stimulus frequencies, in tensor target order: 9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75 Hz.
- Event labels are implicit in target tensor order/frequency list; there is no documented separate event channel.
- License: the upstream repository does not state a license; treat as **unknown / permission required** until clarified by the authors.

## Provenance

Source repository: https://github.com/mnakanishi/12JFPM_SSVEP/tree/master/data
Paper DOI: 10.1371/journal.pone.0140703
