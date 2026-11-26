"""Build a verified formal Windows package into a NEW output folder.

Never installs, deletes an old installation, copies participant data, or opens hardware.
Set PYTHONPATH to the explicitly validated local build-tools directory if required.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurostation_contract import PRODUCT_VERSION, PRODUCT_SEMVER, CONTINUOUS_SSVEP_METHOD


def run(args, env):
    subprocess.run([str(x) for x in args], cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--openbci-runtime", required=True, type=Path)
    a = p.parse_args()
    output = a.output.resolve()
    if output.exists():
        p.error("output must not exist; old builds are never overwritten")
    runtime = a.openbci_runtime.resolve()
    if not (runtime / "OpenBCI_GUI.exe").is_file():
        p.error("explicit OpenBCI runtime is incomplete")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("NEUROSTATION_RUNTIME_CONFIG", "NEUROSTATION_QUALITY_ROOT"):
        env.pop(key, None)
    gpu_runtime = Path(sys.prefix) / "ssvep-directml"
    if gpu_runtime.is_dir():
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(gpu_runtime), env.get('PYTHONPATH'))))
    # Do not resolve unrelated toolchain DLLs through inherited PATH.
    base = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.base_prefix)"], text=True).strip()
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), base, os.environ["SystemRoot"] + "/System32", os.environ["SystemRoot"]])
    run([sys.executable, "-B", ROOT / "scripts/validate_release_configs.py"], env)
    output.mkdir(parents=True)
    artifact = output / "dist/NeuroStation.dist"
    info = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
               for path in paths if path and (ROOT / path).is_file()}
    args = [sys.executable, "-B", "-m", "PyInstaller", "--noconfirm", "--onedir", "--console",
            "--hide-console", "hide-early", "--name", "workstation", "--contents-directory", "_internal",
            "--distpath", output / "dist", "--workpath", output / "work", "--specpath", output,
            "--paths", ROOT, "--hidden-import", "eeg_tools.quality.preview",
            "--collect-all", "brainflow", "--collect-data", "scipy"]
    if gpu_runtime.is_dir():
        args.extend(["--paths", gpu_runtime])
    args.extend(["--collect-binaries", "onnxruntime"])
    for module in ("torch", "matplotlib", "pandas", "pytest", "tkinter", "IPython"):
        args.extend(["--exclude-module", module])
    for source, target in (("configs", "configs"), ("assets", "assets"),
                           ("apps/workstation_ui/locales", "apps/workstation_ui/locales"),
                           ("integrations/openbci_gui/upstream.lock.json", "integrations/openbci_gui"),
                           ("scripts", "scripts"), ("eeg_tools", "eeg_tools"),
                           ("neurostation_contract.py", "."), ("neurostation_signal.py", ".")):
        args.extend(["--add-data", str(ROOT / source) + ";" + target])
    args.append(ROOT / "workstation.py")
    run(args, env)
    (output / "dist/workstation").rename(artifact)
    shutil.copytree(runtime, artifact / "_internal/integrations/openbci_gui/runtime/windows")
    shutil.copy2(ROOT / f"release/NeuroStation-{PRODUCT_VERSION}-Windows-x64.md", artifact / "RELEASE_NOTES.md")
    shutil.copy2(ROOT / f"release/NeuroStation-{PRODUCT_VERSION}-Windows-x64.md", artifact / "UPDATE_NOTES.md")
    run([sys.executable, "-B", ROOT / "scripts/generate_sbom.py", "-o", artifact / "SBOM.json"], env)
    (artifact / "VERSION.txt").write_text(
        f"NeuroStation\nVersion: {PRODUCT_VERSION}\nSemantic version: {PRODUCT_SEMVER}\n"
        f"Build date: {datetime.now(timezone.utc).isoformat()}\nSource commit: {info}\n"
        "Release type: formal standalone portable Windows x64\n", encoding="utf-8")
    # No environment or user-path overrides in the installed launcher.
    (artifact / "Start-NeuroStation.cmd").write_text(
        '@echo off\r\nstart "" "%~dp0workstation.exe" %*\r\n', encoding="ascii")
    (artifact / "Start-EEG-Processing.cmd").write_text(
        '@echo off\nstart "" "%~dp0workstation.exe" --processing-page %*\n', encoding="ascii")
    manifest = {"source_commit": info, "source_files": sources,
                "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip()),
                "version": PRODUCT_VERSION, "built_at": datetime.now(timezone.utc).isoformat(),
                "hardware_validation": "not_run", "four_condition_real_capture": "disabled",
                "sbom_scope": "build-environment inventory, not proof every listed distribution is bundled"}
    (artifact / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    run([sys.executable, "-B", ROOT / "scripts/verify_quality_package.py", "--artifact", artifact,
         "--entry-name", "workstation.exe", "--output", output / "verification"], env)
    run([sys.executable, "-B", ROOT / "scripts/smoke_packaged.py", artifact / "workstation.exe"], env)
    gpu_env = {k: v for k, v in env.items() if k != 'PYTHONPATH'}
    gpu_check = subprocess.run([str(artifact / 'workstation.exe'), '--ssvep-model', 'check', CONTINUOUS_SSVEP_METHOD],
                               cwd=artifact, env=gpu_env, capture_output=True, timeout=90)
    (output / 'verification/ssvep-gpu.json').write_bytes(gpu_check.stdout)
    if gpu_check.returncode:
        raise RuntimeError(f'Packaged GPU check failed: {gpu_check.stdout!r} {gpu_check.stderr!r}')
    changed = [p for p, digest in sources.items() if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"source changed during build: {changed}")
    archive = output / f"NeuroStation-{PRODUCT_VERSION}-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in artifact.rglob("*"):
            if f.is_file():
                z.write(f, str(Path(artifact.name) / f.relative_to(artifact)))
    print(json.dumps({"artifact": str(artifact), "archive": str(archive),
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "verified": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
