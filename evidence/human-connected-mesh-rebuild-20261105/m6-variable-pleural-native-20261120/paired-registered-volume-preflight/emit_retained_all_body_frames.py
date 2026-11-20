"""Emit the exact Float32 all-body frame capture consumed by the native importer."""
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

pose_path, donor_path, destination = map(Path,sys.argv[1:])
with pose_path.open("rb") as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == "0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a"
with donor_path.open() as stream:
    donors = {int(row[1]):[float(value) for value in row[2:]]
              for row in list(csv.reader(stream))[1:]}
pose = json.loads(pose_path.read_text())
assert len(pose["all_bodies"]) == 157 and len(donors)==86

def f32(x):
    return struct.unpack("<f", struct.pack("<f",x))[0]

with destination.open("w",newline="") as stream:
    writer = csv.writer(stream,lineterminator="\n")
    for index, row in enumerate(pose["all_bodies"]):
        assert index == row["body_index"]
        values = []
        for phase in ("current","rest"):
            matrix = row[phase]["local_to_world_row_major_f32"]
            assert len(matrix) == 9 and all(f32(x)==x for x in matrix)
            assert len(row[phase]["position_world_m"]) == 3
            values.extend(f32(x) for x in row[phase]["position_world_m"])
            values.extend(matrix)
        if index in donors:
            assert values == donors[index]
        writer.writerow([index,*values])
