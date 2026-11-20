"""Load the tested Float32 fine candidate and exact topology operation lineage."""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(__file__).resolve().parent
PRIOR = ROOT / "resting-clearance-and-atrium-010"
COMPILER = ROOT / "captured-fat-sliver-495/proposal/compiler"

def load(*, include_interface_polish=True):
    with np.load(PRIOR / "fine-transfer-screen-map.npz", allow_pickle=False) as data:
        original = data["old_world_m"].astype(np.float32)
        candidate = (data["old_world_m"]+data["fine_node_delta_m"]).astype(np.float32)
        roles = data["roles"]
    with np.load(WORK / "fine-determinant-polish.npz", allow_pickle=False) as fit:
        candidate[fit["changed_rows"]] = fit["positions_f32"]
    rows = np.loadtxt(COMPILER/"volume-export/volume/mesh.ele",skiprows=1,dtype=np.int32)
    assert np.array_equal(rows[:,0],np.arange(1,len(rows)+1))
    topo = rows[:,1:5]-1
    material = rows[:,5]
    origins = np.arange(len(topo),dtype=np.int32)
    changed={}
    for stage in (PRIOR/"fine-fixed-boundary-flips.json",WORK/"fine-relaxed-flips.json"):
        operations=json.loads(stage.read_text())["replacements"]
        keep=np.ones(len(topo),dtype=bool)
        additions,labels,ids=[],[],[]
        for operation in operations:
            parent=np.asarray(operation["retired"])
            assert np.all(keep[parent])
            keep[parent]=False
            label=operation["material_id"]
            assert np.all(material[parent]==label)
            ancestors=set()
            for item in origins[parent]:
                if item>=0:
                    ancestors.add(int(item))
                else:
                    ancestors.update(changed[int(item)])
            group=-(len(changed)+1)
            changed[group]=sorted(ancestors)
            additions.extend(operation["new"])
            labels.extend([label]*len(operation["new"]))
            ids.extend([group]*len(operation["new"]))
        topo=np.vstack((topo[keep],np.asarray(additions,dtype=np.int32)))
        material=np.r_[material[keep],np.asarray(labels,dtype=np.int32)]
        origins=np.r_[origins[keep],np.asarray(ids,dtype=np.int32)]
    if include_interface_polish:
        receipt = json.loads((WORK / "fine-interface-polish.json").read_text())
        with np.load(WORK / "fine-interface-polish.npz", allow_pickle=False) as fit:
            candidate[fit["changed_rows"]] = fit["positions_f32"]
        group = -(receipt["reversed_initial_flip"] + 1)
        restored = np.asarray(receipt["restored_source_rows"], dtype=np.int32)
        assert np.array_equal(restored, changed[group])
        keep = origins != group
        assert np.count_nonzero(~keep) == 3 and not np.any(np.isin(restored, origins))
        topo = np.vstack((topo[keep], rows[restored, 1:5] - 1))
        material = np.r_[material[keep], rows[restored, 5]]
        origins = np.r_[origins[keep], restored]
        del changed[group]
    return original,candidate,topo,material,origins,changed,roles,rows[:,1:5]-1,rows[:,5]
