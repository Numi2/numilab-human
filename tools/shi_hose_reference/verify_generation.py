"""Verify pinned source generation, including rejection of a changed import."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADERS = ["shi_hose_reference_generated.hpp", "shi_hose_observables.hpp"]


def generate(source, output, expect_success=True):
    result = subprocess.run([sys.executable, str(HERE / "generate_reference.py"),
                             "--source-dir", str(source), "--output-dir", str(output)],
                            text=True, capture_output=True)
    if expect_success:
        if result.returncode:
            raise RuntimeError(result.stderr)
        subprocess.run([sys.executable, str(HERE / "generate_observables.py"),
                        "--output-dir", str(output)], check=True, capture_output=True)
    elif result.returncode == 0 or "Pinned source digest mismatch" not in result.stderr:
        raise RuntimeError("Changed source import did not fail closed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--native-header-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="shi-hose-generation-") as work:
        root = Path(work)
        generate(args.source_dir, root / "one")
        generate(args.source_dir, root / "two")
        for name in HEADERS + ["flattened-source.json", "native-state-mapping.json"]:
            if (root / "one" / name).read_bytes() != (root / "two" / name).read_bytes():
                raise RuntimeError("Nondeterministic source generation: " + name)
        if args.native_header_dir:
            for name in HEADERS:
                if (root / "one" / name).read_bytes() != (args.native_header_dir / name).read_bytes():
                    raise RuntimeError("Native source header differs from regenerated source: " + name)
        shutil.copytree(args.source_dir, root / "changed")
        imported = root / "changed" / "EAtrium.cellml"
        imported.write_bytes(imported.read_bytes() + b"\n<!-- changed source -->\n")
        generate(root / "changed", root / "rejected", expect_success=False)
        print(json.dumps({"schema": "NumiHuman.CellML-generation-check.v1", "status": "pass",
                          "deterministic": True, "changed_import_rejected": True,
                          "native_headers_checked": bool(args.native_header_dir),
                          "headers": {name: hashlib.sha256((root / "one" / name).read_bytes()).hexdigest()
                                      for name in HEADERS}}, indent=2))


if __name__ == "__main__":
    main()
