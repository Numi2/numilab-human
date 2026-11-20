"""Update only authenticated carrier witnesses whose receiver centroid moved."""
import csv
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"resting-clearance-and-atrium-016"))
from screen_paired_mass_routes import sha

floor = root / "absolute-volume-repaired-tetgen"
geometry = json.loads((floor/"receipt.json").read_text())
assert sha(floor/"mesh.node") == geometry["mesh_node_sha256"]
assert sha(floor/"mesh.ele") == geometry["mesh_ele_sha256"]
old_artifacts = json.loads((root/"paired-native-flow-support-audit.json").read_text())
old_packets = json.loads((root/"paired-carrier-packet-audit/receipt.json").read_text())
limit = json.loads((root/"archived-compensation-independent-check.json").read_text())["transport_distance_limit_m"]
tet = np.loadtxt(floor/"mesh.ele",skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1
points = np.loadtxt(floor/"mesh.node",skiprows=1,usecols=(1,2,3),dtype=np.float32)
moved = json.loads((root/"absolute-volume-surface-delta.json").read_text())["moved_nodes"]
affected = np.flatnonzero(np.any(np.isin(tet,moved),axis=1))
centroids = {row:np.sum(points[tet[row]].astype(np.float64)*.25,axis=0) for row in affected}
out = root/"paired-floor-flow-witness"
out.mkdir(exist_ok=True)
result = []
for old_witness,old_packet in zip(old_artifacts["generated_witnesses"],old_packets["reports"]):
    role = old_packet["role"]
    witness = Path(old_witness["path"])
    packet = Path(old_packet["current_packet_path"])
    assert sha(witness) == old_witness["sha256"]
    assert sha(packet) == old_packet["current_packet_sha256"]
    with np.load(packet) as stream:
        data = dict(stream)
    if "regional_compensation_child_rows" in data:
        assert not np.intersect1d(affected,data["regional_compensation_child_rows"]).size
    replacement = out/witness.name
    changed = primary = 0
    with gzip.open(witness,"rt",newline="") as inp,replacement.open("wb") as buf:
        with gzip.GzipFile(fileobj=buf,filename="",mode="wb",mtime=0,compresslevel=5) as zipped:
            with io.TextIOWrapper(zipped,newline="") as text:
                header = next(inp)
                columns = next(csv.reader([header]))
                cols = {name:columns.index(name) for name in columns}
                row_index = cols["child_global_row"]
                origin_index = cols["source_x_m"]
                centroid_index = cols["child_centroid_x_m"]
                dist_index = cols["distance_m"]
                text.write(header)
                for line in inp:
                    child = int(line.split(",",row_index+1)[row_index])
                    if child not in centroids:
                        text.write(line)
                        continue
                    row = next(csv.reader([line]))
                    source = np.array(row[origin_index:origin_index+3],float)
                    center = centroids[child]
                    distance = float(np.linalg.norm(source-center))
                    assert np.isfinite(distance) and distance <= limit
                    row[centroid_index:centroid_index+3] = [repr(float(x)) for x in center]
                    row[dist_index] = repr(distance)
                    csv.writer(text,lineterminator="\n").writerow(row)
                    changed += 1
                    if row[0] == "carrier":
                        index = int(row[cols["packet_carrier_row"]])
                        assert data["carrier_target_child_rows"][index] == int(row[cols["child_local_row"]])
                        data["carrier_target_distance_m"][index] = distance
                        primary += 1
    new_packet = out/packet.name
    if primary:
        np.savez_compressed(new_packet,**data)
    else:
        new_packet = packet
    result.append(dict(role=role,witness_path=str(replacement),witness_sha256=sha(replacement),
                       carrier_path=str(new_packet),carrier_sha256=sha(new_packet),
                       changed_witness_rows=changed,changed_primary_rows=primary))
report = {"changed_centroids":len(affected),"roles":result,"regional_compensation_scope_untouched":True}
(out/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
