"""Production CLI: fixed workspace saved JSON inputs, new JSON + Markdown outputs."""
from pathlib import Path
import argparse
import sys

WORKSPACE = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(WORKSPACE))
from eeg_tools.ssvep_saved_audit_v5 import AuditError, audit_workspace, publish_reports, validate_output_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, help="New workspace .json path (existing parent)")
    parser.add_argument("--output-markdown", required=True, help="New workspace .md path (existing parent)")
    args = parser.parse_args(argv)
    try:
        # Reject unsafe/existing outputs before reading any input report.
        validate_output_paths(WORKSPACE, args.output_json, args.output_markdown)
        result = audit_workspace(WORKSPACE)
        paths = publish_reports(WORKSPACE, result, args.output_json, args.output_markdown)
    except (AuditError, OSError) as exc:
        print(f"v5 saved audit rejected: {exc}", file=sys.stderr)
        return 2
    print(f"JSON: {paths[0]}\nMarkdown: {paths[1]}")
    print(f"audit_consistent={result['audit_consistent']} research_acceptance={result['research_acceptance']} "
          f"deployment={result['deployment']} runtime_enabled=False production_validated=False")
    # Research refusal is distinct from an invalid/inconsistent audit.
    return 0 if result["audit_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
