# NeuroStation 1.01 Windows x64

## Release information

- Product version: `1.01`
- Semantic version: `1.0.1`
- Release date: `2026-09-23`
- Release form: Windows standalone portable build
- Target system: Windows 10/11 x64
- Release tag: `v1.0.1`

## Updates

- Dataset preview shows every data row from the selected source file;
- all raw columns remain available in original order, including EEG, other auxiliary, analog, timestamp, and marker fields;
- the read-only table uses a virtualized Qt model for long recordings;
- existing timestamp precision restoration and sample-index ordering are retained.

## Included

- PySide6 desktop workstation with Chinese and English UI;
- OpenBCI Cyton hardware acquisition entrypoint and SSVEP four-target workflow;
- hardware preflight, channel configuration validation, session metadata and quality reports;
- OpenBCI recording import and complete read-only dataset review;
- transparent offline FFT analysis, SciPy denoising, diagnostics, and packaged SBOM;
- portable `NeuroStation.dist` directory and `NeuroStation-1.01-Windows-x64.zip` archive.

## Verification

- Core, integration, and UI acceptance tests run by the release workflow;
- Windows standalone package smoke test completes before Release publication;
- hardware and optical/marker timing acceptance remain separate on-device checks.

## Use

After extracting the archive, run:

```powershell
.\NeuroStation.dist\workstation.exe
.\NeuroStation.dist\workstation.exe --diagnostics
```

The production acquisition path requires real OpenBCI Cyton hardware. Without hardware, the application can inspect saved data, import OpenBCI recordings and run diagnostics; it does not generate simulated EEG for production acquisition.

This software is for research and teaching validation and is not a medical device. Verify the channel map, electrode connections, display refresh timing, optical onset, and marker timing before experiments.
