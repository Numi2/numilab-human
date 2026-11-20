"""Copy closed native evidence off temporary storage with byte verification."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess

RUN_FILES = (
    "launch.json", "execution.json", "initial-skin-fem.csv", "skin-bone-witness.csv",
    "stdout.log", "stderr.log", "native/resting-coupled.csv",
    "native/resting-body-motion-partition.csv",
    "native/resting-com-momentum-diagnostic.csv", "native/resting-surface-audit.csv",
)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.source / "execution.json").read_text())
    assert receipt["exit"] == 0
    records = []
    for name in RUN_FILES:
        source = args.source / name
        destination = args.destination / name
        expected = digest(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".transfer")
            assert not temporary.exists()
            subprocess.run(["/usr/bin/ditto", "--hfsCompression", "--noclone",
                            str(source), str(temporary)], check=True)
            assert digest(temporary) == expected, name
            temporary.rename(destination)
        assert digest(destination) == expected, name
        records.append({"path": name, "sha256": expected,
                        "size_bytes": source.stat().st_size})
    (args.destination / "evidence-transfer.json").write_text(
        json.dumps({"source": str(args.source), "files": records}, indent=2) + "\n")

if __name__ == "__main__":
    main()
