# NeuroStation 1.03 Windows x64

## Release information

- Product version: `1.03`
- Semantic version: `1.0.3`
- Build date: `2026-09-24`
- Release form: Windows standalone portable build
- Target system: Windows 10/11 x64
- Intended release tag: `v1.0.3`

## New in 1.03

- SSVEP sessions require an explicit left-eye or right-eye selection and automatically append `_左眼` or `_右眼` to the dataset name without duplicating suffixes.
- Both eyes use the primary display: the selected left or right half presents the existing 10/12/15/20 Hz four-target stimulus; the other half stays black. A second display is no longer required.
- Session, protocol, status, event, and frame-timing files record eye side, dataset names, actual screen details, and stimulus-region geometry. Older datasets without eye-side metadata remain readable.

## Changed in 1.03

- Maps left-eye and right-eye stimulation to the left or right half of the primary display while retaining the existing four-target 10/12/15/20 Hz SSVEP protocol.
- Retains the complete, virtualized read-only dataset preview introduced in 1.02.
- Updates the desktop UI, OpenBCI GUI overlay, README, acceptance documentation, SBOM, and GitHub Actions release metadata to `1.03 / 1.0.3`.

## Fixed in 1.03

- Removes the two-display requirement for left/right eye acquisition; one display can now provide both stimulus regions.
- Keeps the non-selected stimulus half black so visual stimulation cannot spill into the inactive region.
- Prevents repeated `_左眼` or `_右眼` suffixes from polluting dataset names.
- Keeps older sessions readable when eye-side or stimulus-region metadata is absent.
- Prevents OpenBCI GUI overlay startup checks from requiring a network call and validates bilingual overlay catalogs.

## Included

- PySide6 desktop workstation with Chinese and English UI;
- OpenBCI Cyton hardware acquisition and four-target SSVEP workflow;
- hardware preflight, channel configuration validation, session metadata and quality reports;
- OpenBCI recording import, offline FFT analysis, SciPy denoising, diagnostics, and packaged SBOM;
- portable `NeuroStation.dist` directory and `NeuroStation-1.03-Windows-x64.zip` archive.

## Verification boundaries

Desktop unit/UI tests and packaged launch/diagnostic smoke checks validate software behavior. They do not replace Cyton hardware-in-the-loop testing, monitor refresh/photodiode timing measurements, or clinical-device validation. This software is for research and teaching validation, not medical diagnosis.

After extracting the archive, run `NeuroStation.dist\workstation.exe`. No separate Python installation is required.
