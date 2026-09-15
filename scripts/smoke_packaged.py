"""Launch a packaged NeuroStation build briefly and require a clean exit."""

from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def packaged_entry() -> Path:
    if sys.platform == "win32":
        return ROOT / "dist" / "NeuroStation.dist" / "workstation.exe"
    if sys.platform == "darwin":
        return (
            ROOT
            / "dist"
            / "NeuroStation.app"
            / "Contents"
            / "MacOS"
            / "workstation"
        )
    return ROOT / "dist" / "NeuroStation.dist" / "workstation.bin"


def main() -> int:
    entry = packaged_entry()
    if not entry.is_file():
        print(f"Packaged entry does not exist: {entry}", file=sys.stderr)
        return 2
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "darwin":
        # The macOS offscreen backend can abort inside the packaged Qt runtime;
        # minimal still avoids a WindowServer dependency for this smoke test.
        environment["QT_QPA_PLATFORM"] = "minimal"
        brainflow_lib = entry.parent / "brainflow" / "lib"
        if brainflow_lib.is_dir():
            library_path = str(brainflow_lib)
            for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
                existing = environment.get(variable, "")
                environment[variable] = os.pathsep.join(
                    value for value in (library_path, existing) if value
                )
    with tempfile.TemporaryDirectory(prefix="neurostation-smoke-") as directory:
        dataset_root = Path(directory) / "中文数据集"
        if sys.platform != "darwin":
            result = subprocess.run(
                [
                    str(entry),
                    "--smoke-test",
                    "--dataset-root",
                    str(dataset_root),
                ],
                env=environment,
                check=False,
                capture_output=True,
                timeout=30,
            )
            if result.returncode:
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end="")
                print(
                    f"Packaged UI smoke test failed with exit code {result.returncode}",
                    file=sys.stderr,
                )
                return result.returncode
        diagnostics = subprocess.run(
            [str(entry), "--diagnostics", "--dataset-root", str(dataset_root)],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if diagnostics.returncode:
            print("Packaged diagnostics failed", file=sys.stderr)
            return diagnostics.returncode
        diagnostics_value = json.loads(diagnostics.stdout)
        resources = diagnostics_value.get("resources", {})
        configuration = diagnostics_value.get("configuration", {})
        openbci = diagnostics_value.get("openbci_gui", {})
        dependencies = diagnostics_value.get("dependencies", {})
        openbci_runtime_required = sys.platform == "win32"
        if not (
            resources.get("protocol_ready")
            and resources.get("locales_ready")
            and configuration.get("status") in {"ready", "draft"}
            and not configuration.get("error")
            and (openbci.get("executable_ready") or not openbci_runtime_required)
            and dependencies.get("PySide6", {}).get("available")
            and dependencies.get("brainflow", {}).get("available")
        ):
            print(f"Packaged resources are incomplete: {diagnostics_value}", file=sys.stderr)
            return 6
        sbom_path = entry.parent / "SBOM.json"
        if not sbom_path.is_file():
            print(f"Packaged SBOM is missing: {sbom_path}", file=sys.stderr)
            return 10
        try:
            sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("Packaged SBOM is not valid JSON", file=sys.stderr)
            return 11
        if sbom.get("bomFormat") != "CycloneDX" or not sbom.get("components"):
            print("Packaged SBOM is incomplete", file=sys.stderr)
            return 12
        if os.environ.get("NEUROSTATION_SKIP_PACKAGED_WORKER") == "1":
            print(
                "Skipping packaged BrainFlow worker smoke test on this CI platform; "
                "the unbundled synthetic worker acceptance still covers native acquisition.",
                file=sys.stderr,
            )
        else:
            worker = subprocess.run(
                [
                    str(entry),
                    "--acquisition-worker",
                    "--protocol", str(entry.parent / "configs" / "protocols" / "ssvep_four_target_v2.json"),
                    "--output-root", str(dataset_root),
                    "--participant", "PACKAGED-CI",
                    "--session-name", "synthetic-worker-smoke",
                    "--board", "synthetic",
                    "--headless",
                    "--repetitions", "1",
                    "--stimulus-seconds", "0.05",
                    "--rest-seconds", "0.01",
                    "--countdown-seconds", "0.01",
                ],
                env=environment,
                check=False,
                timeout=30,
            )
            if worker.returncode:
                print(f"Packaged worker smoke test failed with exit code {worker.returncode}", file=sys.stderr)
                return worker.returncode
        if os.environ.get("NEUROSTATION_SKIP_PACKAGED_WORKER") == "1":
            return 0
        session_files = list(dataset_root.glob("session_*/session.json"))
        if len(session_files) != 1:
            print("Packaged worker did not create exactly one session", file=sys.stderr)
            return 3
        session = json.loads(session_files[0].read_text(encoding="utf-8"))
        if session.get("status") != "completed" or session.get("recorded_samples_per_channel", 0) <= 0:
            print("Packaged worker session is incomplete", file=sys.stderr)
            return 4
        if (session_files[0].parent / "raw_brainflow.tsv").stat().st_size <= 0:
            print("Packaged worker raw data is empty", file=sys.stderr)
            return 5
        quality_path = session_files[0].parent / "quality.json"
        if not quality_path.is_file():
            print("Packaged worker quality report is missing", file=sys.stderr)
            return 7
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality.get("samples_per_channel", 0) <= 0 or quality.get("channel_count", 0) <= 0:
            print("Packaged worker quality report is incomplete", file=sys.stderr)
            return 8
        manifest_rows = (session_files[0].parent / "manifest.csv").read_text(encoding="utf-8")
        if "quality.json" not in manifest_rows or "ssvep_config.json" not in manifest_rows:
            print("Packaged worker manifest is incomplete", file=sys.stderr)
            return 9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
