"""Retire a scratch MRVPACK only after verifying its lossless recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack",type=Path,nargs="+")
    args=parser.parse_args()
    for item in args.pack:
        path=item.resolve(strict=True)
        expected=digest(path)
        receipt=path.with_suffix(".receipt.json")
        if receipt.is_file():
            assert json.loads(receipt.read_text())["pack_file_sha256"]==expected
        dest=Path(str(path)+".zst")
        assert not dest.exists()
        zstd="/opt/homebrew/bin/zstd"
        subprocess.run([zstd,"-T2","-3","--check",str(path),"-o",str(dest)],check=True)
        with subprocess.Popen([zstd,"-dc",str(dest)],stdout=subprocess.PIPE) as p:
            assert hashlib.file_digest(p.stdout,"sha256").hexdigest()==expected
            assert p.wait()==0
        report={"sha256":expected,"uncompressed_bytes":path.stat().st_size,
                "archived_bytes":dest.stat().st_size,
                "restore_command":[zstd,"-d",str(dest),"-o",str(path)]}
        Path(str(dest)+".recovery.json").write_text(json.dumps(report,indent=2)+"\n")
        assert digest(path)==expected
        path.unlink()
        print(json.dumps(report),flush=True)

if __name__=="__main__":
    main()
