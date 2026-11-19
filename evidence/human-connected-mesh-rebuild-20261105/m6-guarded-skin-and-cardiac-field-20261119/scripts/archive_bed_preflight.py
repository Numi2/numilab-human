"""Compress only this experiment's generated preflight evidence after recovery checks."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

from run_segment_control import HERE, digest

ZSTD="/opt/homebrew/bin/zstd"
KINDS={"baseline","anchor-pinned","footprint-pinned","full"}

def archive(raw):
    path=raw.with_suffix(raw.suffix+".zst")
    expected=digest(raw)
    size=raw.stat().st_size
    if not path.exists():
        subprocess.run([ZSTD,"-3","--quiet","-o",str(path),str(raw)],check=True)
    with subprocess.Popen([ZSTD,"-dc",str(path)],stdout=subprocess.PIPE) as proc:
        actual=hashlib.file_digest(proc.stdout,"sha256").hexdigest()
        assert proc.wait()==0 and actual==expected
    check=path.with_suffix(path.suffix+".recovery.json")
    check.write_text(json.dumps(dict(sha256=expected,bytes=size,archive_bytes=path.stat().st_size,
        verified_by="streamed decompression and SHA-256",restore_command=[ZSTD,"-d",str(path),"-o",str(raw)]),indent=2)+"\n")
    raw.unlink()
    print(raw.name,expected,flush=True)

def main():
    for kind in sys.argv[1:]:
        assert kind in KINDS
        full=kind=="full"
        root=HERE/("native-segment32-control-footprint" if full else f"bed-preflight-{kind}")/"native"
        for path in (root/"resting-human.mrvpack",root/"skin-source-influences.bin",
                     root/"accepted-geometry"/f"step-{8000 if full else 1}.mrvpack"):
            if path.is_file():
                archive(path)

if __name__=="__main__":
    main()
