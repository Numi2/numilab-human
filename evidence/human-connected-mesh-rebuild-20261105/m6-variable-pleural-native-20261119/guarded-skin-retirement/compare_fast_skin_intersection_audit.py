import argparse
import json
from pathlib import Path
import runpy
import sys
import time

from fast_skin_intersection_audit import install_for_float32

parser=argparse.ArgumentParser()
parser.add_argument("pack",type=Path)
parser.add_argument("previous",type=Path)
parser.add_argument("candidate",type=Path)
args=parser.parse_args()
audit_path=Path(__file__).resolve().parent.parent/"resting-surface-clearance-001/audit.py"
install_for_float32()
before=time.monotonic()
runpy.run_path(str(audit_path))["run"](args.pack.resolve(),args.candidate.resolve())
elapsed=time.monotonic()-before
p,q=(json.loads(f.read_text()) for f in (args.previous,args.candidate))
assert p==q
print(dict(whole_report_exact_parity=True,wall_seconds=elapsed,
           cross_pairs=sum(q["crossing_pairs_by_semantic"].values()),
           self_pairs=q["self_intersections"]["count"]),flush=True)
