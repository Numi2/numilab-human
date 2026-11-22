"""Recover the exact row experiment and its verified parent archives."""
import sys
from pathlib import Path
from archive_cavity_trial import archive_files, sha


def save(case):
    keys = ("build.py", "capture.inc", "verify_row.py", "archive_row.py", "result.json",
            "owner/identities.json", "owner/numi-human-native", "build/libmetalrobo.dylib",
            "parent/launch.json", "one/launch.json", "one/execution.json", "one/failure.json",
            "one/stdout.log", "one/stderr.log")
    files = {k: case / k for k in keys}
    for p in sorted((case / "one/native-traces").glob("*.csv")):
        files[str(p.relative_to(case))] = p
    files["parent-wrench-recovery.json"] = case.parent / "recovery.json"
    name = "registered-bed-row-" + sha(case / "result.json")[:16] + ".tar.zst"
    archive_files(files, name, case / "recovery.json",
                  "Actual assembled FEM impulse and geometric contact witness at the same whole-body "
                  "Newton trial. The parent receipt identifies all unchanged sources and geometry.")


if __name__ == "__main__":
    save(Path(sys.argv[1]).resolve())
