"""Pin projection tables to the current paired parent without changing their rows."""
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"resting-clearance-and-atrium-016"))
from screen_paired_mass_routes import COMPILER, paired_reference_carriers, sha

old = root/"complete-fine-p1"
old_state = old/"current-paired-volume-state.json"
state = json.loads(old_state.read_text())
ledger = root/"paired-native-parent-ledger"
receipt = json.loads((ledger/"receipt.json").read_text())
if sha(old_state) != receipt["input_state_sha256"]:
    raise ValueError("Paired ledger references a different P1 source state")
for name, expected in receipt["files"].items():
    if sha(ledger/name) != expected:
        raise ValueError(f"Current parent ledger file changed: {name}")
previous = state["source330_identity"]
paths = {key:Path(pin["path"]) for key,pin in previous.items()}
rest, tets, expected_masses, _ = paired_reference_carriers(
    COMPILER,paths["source330_nodes"],paths["source330_tets"])
previous_nodes = np.loadtxt(paths["source330_nodes"],delimiter=",",skiprows=1)
previous_tets = np.loadtxt(paths["source330_tets"],delimiter=",",skiprows=1)
base = ledger/"compiled-parent-cell-mass.csv"
new_paths = {key:Path(str(base)+suffix) for key,suffix in
             (("source330_nodes",".nodes.csv"),("source330_tets",".tets.csv"),
              ("source330_mass",""),("source330_labels",".labels.csv"))}
new_nodes = np.loadtxt(new_paths["source330_nodes"],delimiter=",",skiprows=1)
new_tets = np.loadtxt(new_paths["source330_tets"],delimiter=",",skiprows=1)
new_masses = np.loadtxt(base,delimiter=",",skiprows=1)
expected_points = previous_nodes[:,2:5] + (rest-previous_nodes[:,5:8])
if (not np.array_equal(new_nodes[:,2:5], expected_points) or
    not np.array_equal(new_nodes[:,[0,1,9]],previous_nodes[:,[0,1,9]]) or
    not np.array_equal(new_tets[:,:7],previous_tets[:,:7]) or
    not np.array_equal(new_tets[:,3:7],tets) or
    not np.array_equal(new_masses[:,2],expected_masses) or
    sha(new_paths["source330_labels"]) != previous["source330_labels"]["sha256"]):
    raise ValueError("Paired ledger differs from the P1 projection frame or changes source identities")

out = root/"complete-fine-p1-current-parent"
out.mkdir()
state_path = out/old_state.name
state["source330_identity"] = {key:{"path":str(path),"sha256":sha(path)}
                               for key,path in new_paths.items()}
state["archived_source_identity"] = previous
state["paired_ledger_receipt_sha256"] = sha(ledger/"receipt.json")
state["projection_frame_unchanged"] = True
state_path.write_text(json.dumps(state,indent=2)+"\n")
state_sha = sha(state_path)
for role in ("skin","fat"):
    info = json.loads((old/f"{role}-p1-receipt.json").read_text())
    result = out/role.title()
    result.mkdir()
    new_base_hash = None
    for path_key, hash_key in (("p1_path","p1_sha256"),
                               ("p1_one_hop_path","p1_one_hop_sha256")):
        path = Path(info[path_key])
        if sha(path) != info[hash_key]:
            raise ValueError(f"Unverified original P1 artifact: {path}")
        with np.load(path,allow_pickle=False) as f:
            data = {key:f[key] for key in f.files}
        if (str(data["current_state_sha256"]) != sha(old_state) or
            str(data["source330_mass_sha256"]) != previous["source330_mass"]["sha256"]):
            raise ValueError("Previous P1 state or ancestry pin differs from archive")
        data["current_state_path"] = np.asarray(str(state_path))
        data["current_state_sha256"] = np.asarray(state_sha)
        data["source330_mass_sha256"] = np.asarray(sha(base))
        if new_base_hash is not None:
            if str(data["admissibility_source_support_sha256"]) != original_base_hash:
                raise ValueError("One-hop pin does not reference its base P1 packet")
            data["admissibility_source_support_sha256"] = np.asarray(new_base_hash)
        output = result/path.name
        np.savez_compressed(output,**data)
        with np.load(output,allow_pickle=False) as verified:
            for key in data.keys() - {"current_state_path","current_state_sha256",
                                     "source330_mass_sha256","admissibility_source_support_sha256"}:
                if not np.array_equal(verified[key],data[key],equal_nan=False):
                    raise ValueError(f"Altered geometry or P1 projection array {key}")
        if new_base_hash is None:
            new_base_hash = sha(output)
            original_base_hash = info[hash_key]
        info[path_key],info[hash_key] = str(output),sha(output)
    info["state_path"],info["state_sha256"] = str(state_path),state_sha
    info["projection_frame_unchanged"] = True
    (out/f"{role}-p1-receipt.json").write_text(json.dumps(info,indent=2)+"\n")
print(f"Both role P1 tables rebound to {sha(base)}, geometry and projected ancestry unchanged")
