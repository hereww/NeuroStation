# OpenBMI SSVEP (Lee2019) — verified subset

## Result

A **complete, checksum-verified one-subject subset** was downloaded: subject 1, session 1 and session 2 SSVEP MAT files. This is not the full 54-subject OpenBMI SSVEP release; no claim of full-dataset download is made.

## Task-relevant facts

- OpenBMI contains 54 subjects and 2 sessions for SSVEP; the downloaded files are the SSVEP files for subject 1 in both sessions.
- Original acquisition is 1000 Hz with 62 EEG channels; the released/analysis data are documented as downsampled to 100 Hz.
- Four target frequencies/classes: 12.0, 8.57, 6.67, and 5.45 Hz. The training phase has labels; the online test phase is reported as unlabeled for classification use, so inspect each file's train/test fields before evaluation.
- License: CC0-1.0 as listed in the MOABB/GigaDB metadata.

## Access limitation / candidate note

The public object store is directly readable, but the complete release is much larger than this verified subset. The manifest intentionally records only the two local files; expand subjects only as a separate acquisition task.

## Provenance

GigaDB DOI: https://doi.org/10.5524/100542
Object-store root used by the public downloader: `https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/`
