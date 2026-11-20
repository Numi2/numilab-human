"""Emit the current reference in the existing native parent CSV layout."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"resting-clearance-and-atrium-016"))
from screen_paired_mass_routes import COMPILER, paired_reference_carriers, sha

p1 = root/"complete-fine-p1"
state_file = p1/"current-paired-volume-state.json"
state = json.loads(state_file.read_text())
paths = {k:Path(v["path"]) for k,v in state["source330_identity"].items()}
for name, path in paths.items():
    if sha(path) != state["source330_identity"][name]["sha256"]:
        raise ValueError(f"Archived parent identity changed: {name}")
frame_path = root/"paired-native-source-frame-audit.json"
frame = json.loads(frame_path.read_text())
if frame["state_sha256"] != sha(state_file) or frame["historical_rest_alignment_max_m"] > 2e-12:
    raise ValueError("Current material reference frame has not been established")
old = np.loadtxt(paths["source330_nodes"],delimiter=",",skiprows=1)
tet_table = np.loadtxt(paths["source330_tets"],delimiter=",",skiprows=1)
archive_rest, connectivity, masses, _ = paired_reference_carriers(
    COMPILER, paths["source330_nodes"],paths["source330_tets"])
previous = root.parent/"resting-clearance-and-atrium-010"
paired = previous/"paired-skin-fat-refit-001/paired-geometry-and-mass.npz"
registration = previous/"paired-registration-with-face-ancestry/registration.npz"
for path in (paired,registration):
    if sha(path) != state["current_reference"]["inputs_sha256"][str(Path("..")/path.relative_to(root.parent))]:
        raise ValueError(f"Paired source changed: {path}")
with np.load(paired,allow_pickle=False) as packet:
    reference = packet["points_world_m"].reshape(-1,3)
with np.load(registration,allow_pickle=False) as packet:
    local = packet["physical_layers_donor_local_m"].reshape(-1,3)
    owners = np.tile(packet["donor_body"],3)
if not np.array_equal(owners,old[:,9]):
    raise ValueError("Paired reference changed the source node donor")
rest = (reference @ np.asarray(frame["historical_world_to_captured_rest_R"])
        + np.asarray(frame["historical_world_to_captured_rest_t"]))
current = old[:,2:5] + (archive_rest-old[:,5:8])
delta = rest[connectivity[:,1:]] - rest[connectivity[:,:1]]
volume = np.einsum("ni,ni->n",delta[:,0],np.cross(delta[:,1],delta[:,2]))/6.
material = tet_table[:,1].astype(int)
density = np.select((material==15,material==16),(1000.,900.))
if np.any(volume <= 0.) or np.max(np.abs(volume*density-masses)) > 2e-15:
    raise ValueError("Paired native reference changed orientation or derived material mass")
# Match the existing native accounting owner: a quarter of each cell's mass
# belongs to its corner's persisted donor and material.
split = np.zeros((len(owners),2))
np.add.at(split,(connectivity.ravel(),np.repeat(material-15,4)),np.repeat(masses/4.,4))
nodal = split.sum(axis=1).astype("<f4")
bits = nodal.view("<u4")
out = root/"paired-native-parent-ledger"
out.mkdir()
base = out/"compiled-parent-cell-mass.csv"
def write(suffix, rows):
    path = Path(str(base)+suffix)
    with path.open("w",newline="") as f:
        writer = csv.writer(f,lineterminator="\n")
        with Path(str(paths["source330_mass"])+suffix).open(newline="") as original:
            writer.writerow(next(csv.reader(original)))
        writer.writerows(rows)
    return path

mass_path = write("",zip(range(len(material)),material,masses))
node_path = write(".nodes.csv",((i,int(old[i,1]),*current[i],*rest[i],float(nodal[i]),
                             owners[i],*local[i],bits[i]) for i in range(len(owners))))
tet_path = write(".tets.csv",((i,material[i],int(tet_table[i,2]),*connectivity[i],
                             volume[i],masses[i]) for i in range(len(material))))
source_path = write(".source.csv",((row,15+offset,owners[row],split[row,offset],*local[row])
                                  for offset in range(2) for row in np.flatnonzero(split[:,offset])))
labels_path = Path(str(base)+".labels.csv")
shutil.copyfile(paths["source330_labels"],labels_path)
new_paths = [mass_path,node_path,tet_path,source_path,labels_path]
report = {
    "scope":"Regenerated paired reference ledger using the existing native CSV layout. The live scene has not captured or adopted this ledger.",
    "parent_rows":len(material),"nodal_rows":len(owners),
    "parent_mass_kg":float(masses.sum()),
    "mass_by_material_kg":{str(i):float(masses[material==i].sum()) for i in (15,16)},
    "node_mass_float32_rounding_max_kg":float(np.abs(nodal-split.sum(axis=1)).max()),
    "carrier_rest_and_native_rest_position_difference_m":float(np.linalg.norm(rest-archive_rest,axis=1).max()),
    "original_typed_label_sha256":state["source330_identity"]["source330_labels"]["sha256"],
    "input_state_sha256":sha(state_file),
    "frame_check_sha256":sha(frame_path),
    "files":{path.name:sha(path) for path in new_paths},
}
(out/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
