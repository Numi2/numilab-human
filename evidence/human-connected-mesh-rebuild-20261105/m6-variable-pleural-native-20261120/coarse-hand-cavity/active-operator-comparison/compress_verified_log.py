"""Store closed benchmark logs with APFS compression and verify exact bytes."""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory


def identity(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise RuntimeError(f"Expected a singly linked regular log: {path}")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_closed(path):
    result = subprocess.run(["/usr/sbin/lsof", "-t", str(path)],
                            capture_output=True, check=False)
    if result.returncode != 1 or result.stdout or result.stderr:
        raise RuntimeError(f"Could not verify log is closed: {path}")


def compress(path):
    assert_closed(path)
    before_id = identity(path)
    checksum = digest(path)
    before_bytes = path.stat().st_blocks * 512
    with TemporaryDirectory(dir=path.parent, prefix=".verify-compression-") as directory:
        copy = Path(directory) / path.name
        subprocess.run(["/usr/bin/ditto", "--hfsCompression", "--noclone",
                        str(path), str(copy)], check=True)
        after_bytes = copy.stat().st_blocks * 512
        if digest(copy) != checksum or copy.stat().st_size != before_id[2]:
            raise RuntimeError(f"Compressed copy differs: {path}")
        assert_closed(path)
        if identity(path) != before_id:
            raise RuntimeError(f"Source changed during compression: {path}")
        if after_bytes < before_bytes:
            os.replace(copy, path)
    if digest(path) != checksum:
        raise RuntimeError(f"Final stored log differs: {path}")
    return {"path": str(path), "sha256": checksum, "size_bytes": before_id[2],
            "allocated_before_bytes": before_bytes,
            "allocated_after_bytes": path.stat().st_blocks * 512}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    with args.evidence.open("x") as receipt:
        for path in args.logs:
            if not (path.parent / "execution.json").is_file():
                raise RuntimeError(f"Run has no terminal receipt: {path.parent}")
            receipt.write(json.dumps(compress(path.resolve())) + "\n")
            receipt.flush()


if __name__ == "__main__":
    main()
