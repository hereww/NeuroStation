"""Generate a small reproducible Python dependency SBOM for NeuroStation."""
from __future__ import annotations

import argparse
import json
from importlib.metadata import distributions
from pathlib import Path


def build_sbom() -> dict[str, object]:
    packages = []
    for distribution in sorted(distributions(), key=lambda item: item.metadata.get("Name", "").lower()):
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name:
            continue
        packages.append({"name": name, "version": version})
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:neurostation:sbom:mvp1.0.1",
        "metadata": {"component": {"name": "NeuroStation", "version": "1.0.1"}},
        "components": [
            {"type": "library", "name": item["name"], "version": item["version"]}
            for item in packages
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_sbom(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
