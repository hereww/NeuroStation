# BCI Competition / BNCI candidate list and access notes

This file records candidates checked during the SSVEP/BCI holdout search that were **not added as downloaded datasets** in this run. No success is implied for entries marked restricted/not suitable.

| Candidate | Relevance | Access / license finding | Why not downloaded here |
|---|---|---|---|
| OpenBMI / Lee2019 SSVEP full release | Direct SSVEP; 54 subjects, 2 sessions, 4 frequencies | Public GigaDB/Wasabi object store; CC0-1.0 metadata | A verified subject-1, sessions-1/2 subset is present in `openbmi_ssvep_lee2019`; full release was intentionally not fetched in this short run. |
| BCI Competition IV Dataset 2a / BNCI2014-001 | Public BCI benchmark, but **motor imagery**, not SSVEP | Public BNCI/MOABB route; not an SSVEP frequency-label dataset | Not suitable for the requested SSVEP holdout; no files downloaded. |
| BCI Competition IV Dataset 2b / BNCI2014-004 | Public BCI benchmark, but **motor imagery**, not SSVEP | Public BNCI/MOABB route; not an SSVEP frequency-label dataset | Not suitable for the requested SSVEP holdout; no files downloaded. |
| BNCI2015-003 Center Speller | Public visual BCI, but **P300/ERP**, not SSVEP | Public BNCI/MOABB route; event labels are ERP/P300-style | Not suitable for SSVEP frequency decoding; no files downloaded. |
| BCI Competition / BNCI SSVEP-specific artifact | Potentially useful if a stable public artifact can be confirmed | No stable, directly downloadable SSVEP artifact was verified in this pass | No guessed URL, login bypass, or fabricated success. |

## Sources checked

- BNCI database landing page: https://bnci-horizon-2020.eu/database/data-sets
- BCI Competition archive: https://www.bbci.de/competition/
- MOABB dataset metadata/code used to cross-check public SSVEP and BNCI dataset descriptions: https://github.com/NeuroTechX/moabb

## Interpretation

The downloaded SSVEP datasets in sibling directories are the reproducible candidates for SSVEP downstream holdout evaluation. The BCI Competition/BNCI entries above are retained as **candidate notes only** because the confirmed public examples are MI/P300 rather than SSVEP, or because no stable SSVEP download artifact was verified.
