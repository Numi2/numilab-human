"""Run a pinned native command with its authored environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

launch, receipts = map(Path,sys.argv[1:3])
d = json.loads(launch.read_text())
for path, expected in d["pins"].items():
    with open(path,"rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==expected, path
settings={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
start=time.monotonic()
with receipts.with_suffix(".stdout").open("wb") as stdout, receipts.with_suffix(".stderr").open("wb") as stderr:
    try:
        code=subprocess.run(d["argv"], cwd=d["cwd"], env=settings|d["environment"],
                            stdout=stdout, stderr=stderr, timeout=285).returncode
    except subprocess.TimeoutExpired:
        code=None
elapsed=time.monotonic()-start
result=dict(exit_code=code,wall_seconds=elapsed)
receipts.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
sys.exit(0 if code==0 else 1)
