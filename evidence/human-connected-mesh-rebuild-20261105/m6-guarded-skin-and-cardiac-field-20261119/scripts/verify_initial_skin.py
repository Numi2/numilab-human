"""Check the held-out initial capture and immutable bed support source rows."""
import csv
import json

import numpy as np

from fit_skin_source import ROOT, PRIOR, BASE, bindings, capture, decode, axes
from trim_skin_witnesses import Pose, projected_guard

def main():
    source=ROOT.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin"
    model=decode(source.read_bytes())
    receipt,points,_=capture(BASE)
    d=np.load(ROOT/"trim-skin-proposal.npz")["displacement_m"]
    active=np.any(d!=0,axis=1)
    assert len(d)==model["nv"]
    assert not np.any(active & (points[:,2]<=.020))
    path=PRIOR/"wider-release-control-16/native/resting-local-bed-sections.csv"
    with path.open() as file:
        anchors={int(row["source_geometry_index"]) for row in csv.DictReader(file)}
    assert len(anchors)>0 and min(anchors)>=0 and max(anchors)<len(d)
    assert not np.any(active[list(anchors)])
    matrix=np.einsum("vb,bij->vij",model["weights"],axes(model,receipt)[0])
    posed=(points+np.einsum("vij,vj->vi",matrix,d)).astype("<f4")
    inventory=ROOT.parent/"resting-surface-clearance-001/initial.json"
    check=Pose(BASE,json.loads(inventory.read_text()),posed,active)
    orientation=projected_guard(check,active)
    report,failures=check.audit(active,check_self=True)
    result=dict(support_rows_unchanged=len(anchors),orientation_failures=len(orientation),
                initial_pairs=report,initial_pack_sha256=receipt["pack_file_sha256"],
                qualification="held-out frozen Float32 initial pose")
    (ROOT/"trim-skin-held-out-initial.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)

if __name__=="__main__":
    main()
