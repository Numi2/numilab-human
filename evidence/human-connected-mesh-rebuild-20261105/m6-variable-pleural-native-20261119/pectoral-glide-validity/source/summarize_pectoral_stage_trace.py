"""Decode the line-search bounds copied from the native GPU command buffer."""
import argparse
import json
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stderr", type=Path)
parser.add_argument("receipt", type=Path)
args = parser.parse_args()
stages = []
with args.stderr.open() as stream:
    for line in stream:
        if line.startswith("vascular_iterate="):
            record = json.loads(line.partition("=")[2])
            arenas = record.pop("arenas")
            record["alpha"] = struct.unpack("<f", bytes.fromhex(arenas["alpha"]))[0]
            record["object_alphas"] = [v[0] for v in struct.iter_unpack(
                "<4f", bytes.fromhex(arenas["fem_line_search"]))]
            record["status"] = struct.unpack_from("<5I", bytes.fromhex(arenas["status"]))
            stages.append(record)
        elif line.startswith("fem_newton_terminal_certificate="):
            record = json.loads(line.partition("=")[2])
            record["stage"] = "mixed_terminal_certificate"
            stages.append(record)
if not stages:
    raise SystemExit("No matching GPU stage records")
args.receipt.write_text(json.dumps(stages, indent=2) + "\n")
