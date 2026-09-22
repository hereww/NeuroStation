# NeuroStation 1.0 Windows x64

## Release information

- Product version: `1.0`
- Semantic version: `1.0.0`
- Release date: `2026-09-22`
- Release form: Windows standalone portable build
- Target system: Windows 10/11 x64
- Release tag: `v1.0.0`

## Included

- PySide6 desktop workstation with Chinese and English UI;
- OpenBCI Cyton hardware acquisition entrypoint;
- SSVEP four-target workflow at 10/12/15/20 Hz;
- hardware preflight, channel configuration validation, session metadata and quality reports;
- OpenBCI recording import and read-only dataset review;
- transparent offline FFT analysis and SciPy denoising outputs;
- diagnostics report, persistent event log and packaged SBOM;
- portable `NeuroStation.dist` directory and `NeuroStation-1.0-Windows-x64.zip` archive.

## Verification

- Core and integration tests passed in the release workflow;
- independent UI tests and UI scale smoke tests passed;
- Windows standalone packaged smoke test passed;
- SBOM and packaged version metadata are generated during the build;
- Linux and macOS acceptance builds are verified separately by the desktop workflow.

## Use

After extracting the archive, run:

```powershell
.\NeuroStation.dist\workstation.exe
.\NeuroStation.dist\workstation.exe --diagnostics
```

The production acquisition path requires real OpenBCI Cyton hardware. Without hardware, the application can inspect saved data, import OpenBCI recordings and run diagnostics; it does not generate simulated EEG for production acquisition.

Before a real experiment, manually verify the channel map, reference and BIAS electrodes, display refresh timing, optical onset and marker timing. This software is for research and teaching validation and is not a medical device.
