"""Pin exactly the native section dependency neighborhood before rechecking skin."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from fit_skin_source import ROOT, PRIOR, decode

def main():
    baseline=ROOT/"bed-preflight-baseline/native"
    candidate=ROOT/"bed-preflight-anchor-pinned/native"
    name="resting-bed-section-footprint.csv"
    base,trial=(np.genfromtxt(p/name,delimiter=",",names=True) for p in (baseline,candidate))
    assert np.array_equal(base["vertex"],np.arange(len(base))) and np.array_equal(base["vertex"],trial["vertex"])
    ids=base["vertex"].astype(int)
    source=decode((ROOT.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin").read_bytes())
    original=np.load(ROOT/"trim-three-poses.npz")["displacement_m"]
    active=np.any(original!=0,axis=1)
    native_before,native_after=(np.stack([a[k] for k in "xyz"],axis=1) for a in (base,trial))
    move=np.linalg.norm(native_before-native_after,axis=1)
    area=(base["tributary_area"]>0)|(trial["tributary_area"]>0)
    assert len(ids)==len(original) and np.isfinite(move).all() and move.max()<.005
    rest_guard=.000005
    # A changed vertex and its region minimum can each move by at most
    # the observed whole-vertex displacement. Include all face neighbors.
    near=(base["z"]<=base["region_min"]+.020+2*move.max()+rest_guard)|area
    pin=np.zeros(len(ids),bool)
    pin[near]=True
    faces=source["inds"]
    pin[faces[np.any(near[faces],axis=1)].ravel()]=True
    section=baseline/"resting-local-bed-sections.csv"
    with section.open() as f:
        seed_ids=sorted({int(r["source_geometry_index"]) for r in csv.DictReader(f)})
    assert len(seed_ids)==32 and min(seed_ids)>0
    pin[seed_ids]=True # prior diagnostic rows
    pin[np.asarray(seed_ids)-1]=True # actual one-based NHCNT source identities
    pin[base["region"]!=trial["region"]]=True
    area_pins=int(np.count_nonzero(pin & active))
    table=Path("/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanWholeBodySkinReduction.hpp")
    body=table.read_text().split("kWholeBodySkinSourceVertices={",1)[1].split("};",1)[0]
    samples=np.unique([int(n) for n in re.findall(r"(?m)\b(\d+)(?:u)?(?=\s*,)",body) if int(n)<len(ids)])
    assert len(samples)>1000
    nearest=base["nearest_thickness_vertex"][base["tributary_area"]>0].astype(int)
    assert np.isin(nearest,samples).all()
    pin[nearest]=True
    for region in np.unique(base["region"]):
        query=ids[(base["tributary_area"]>0)&(base["region"]==region)]
        contenders=samples[active[samples] & (base["region"][samples]==region)]
        if not len(query) or not len(contenders): continue
        owner=base["nearest_thickness_vertex"][query].astype(int)
        gate=np.linalg.norm(native_before[query]-native_before[owner],axis=1)
        dist=np.linalg.norm(native_before[query,None,:]-native_before[contenders][None,:,:],axis=2)
        can_win=np.any(dist<=gate[:,None]+move[contenders][None,:]+rest_guard,axis=0)
        pin[contenders[can_win]]=True
    report=dict(all_protected_rows=int(pin.sum()), area_and_partition_edit_retirements=area_pins,
                edited_rows_overlapping_footprint=int((pin & active).sum()),
                active_rows_after_pin=int((~pin & active).sum()),
                max_observed_native_motion_m=float(move.max()),
                exact_thickness_samples=len(samples),
                skin_patch_guard_m=rest_guard,
                base_footprint_sha256=hashlib.file_digest((baseline/name).open("rb"),"sha256").hexdigest(),
                trial_footprint_sha256=hashlib.file_digest((candidate/name).open("rb"),"sha256").hexdigest(),
                source_seed_identity_convention="NHCNT one-based",
                support_rows_count=len(seed_ids))
    print(report,flush=True)
    np.savez_compressed(ROOT/"bed-section-pins.npz",vertices=ids[pin])
    (ROOT/"bed-section-pins.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
