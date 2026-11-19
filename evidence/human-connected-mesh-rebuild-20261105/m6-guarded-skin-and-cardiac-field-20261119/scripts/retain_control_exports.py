"""Reuse verified byte-identical control archives and preserve restoration receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from run_segment_control import HERE, PRIOR, digest

def main():
    original=PRIOR/"wider-release-control-16/native/accepted-geometry"
    reference=json.loads((original/"step-8000.receipt.json").read_text())
    archive=original/"step-8000.mrvpack.zst"
    expected=reference["pack_file_sha256"]
    zstd="/opt/homebrew/bin/zstd"
    with subprocess.Popen([zstd,"-dc",str(archive)],stdout=subprocess.PIPE) as proc:
        actual=hashlib.file_digest(proc.stdout,"sha256").hexdigest()
        assert proc.wait()==0 and actual==expected
    for key in ("native-segment16-control","native-segment32-control","native-segment32-control-cache"):
        out=HERE/key/"native/accepted-geometry"
        raw=out/"step-8000.mrvpack"
        candidate=json.loads((out/"step-8000.receipt.json").read_text())
        assert candidate["pack_file_sha256"]==expected and digest(raw)==expected
        link=out/archive.name
        os.link(archive,link)
        assert link.samefile(archive)
        report={"sha256":expected,"bytes":raw.stat().st_size,"archive_bytes":link.stat().st_size,
                "verified_by":"decompress and SHA-256 comparison, same inode",
                "restore_command":[zstd,"-d",str(link),"-o",str(raw)]}
        (out/"step-8000.mrvpack.zst.recovery.json").write_text(json.dumps(report,indent=2)+"\n")
        raw.unlink()
        print(key,expected,flush=True)

if __name__=="__main__":
    main()
