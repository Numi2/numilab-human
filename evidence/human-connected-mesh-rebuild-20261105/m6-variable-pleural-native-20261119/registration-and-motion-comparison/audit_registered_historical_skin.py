"""Use common bone IDs to test the archived skin against current native targets."""
import argparse
import json
import mmap
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fit_cardiac_volume_grid import HERE, hash_file
from recover_historical_bone_registration import READER, BONE, rigid_fit
from fast_skin_intersection_audit import install_for_float32

clear=READER["clearance"]
exact=READER["intersections"]
SKIN=READER["SKIN"]


def source_bones(path):
    with np.load(path) as z:
        ids=z["bank_bone"]
        parents=z["bank_body"]
        edges=z["bank_face_offsets"]
        f=z["faces"]
        p=z["points_world_f32"]
    assert len(ids)==185 and len(set(ids.tolist()))==185
    return {int(bone):(int(parent), p[f[lo:hi]].astype(float))
            for bone,parent,lo,hi in zip(ids,parents,edges[:-1],edges[1:])}


def fit_bones(mapped, bodies, bones):
    pairs=[]
    for key,_,p,f in READER["primitives"](mapped,bodies):
        if key[0]!=BONE:
            continue
        _,old=bones[key[1]]
        native=p[f].astype(float)
        assert old.shape==native.shape,(key,old.shape,native.shape)
        pairs.append((old.mean(axis=1),native.mean(axis=1),key[1]))
    assert sorted(k for _,_,k in pairs)==sorted(bones)
    x=np.concatenate([a for a,_,_ in pairs])
    y=np.concatenate([b for _,b,_ in pairs])
    q,t,e=rigid_fit(x,y)
    maximum=float(e.max())
    per_bank={key:float(np.linalg.norm(a@q+t-b,axis=1).max()) for a,b,key in pairs}
    proof=dict(rotation_row=q.tolist(),translation_m=t.tolist(),
               all_bank_max_face_center_error_m=maximum,
               per_bank_max_face_center_error_m=per_bank)
    if maximum>5.e-6:
        raise ValueError(f"historical and current poses need link registration: {maximum} m")
    return q,t,proof


def audit_skin(positions,skin_faces,mapped,bodies):
    skin=clear._exact_surface_records(positions,skin_faces)
    target=exact._prepare_surface_aabb(skin)
    broad=clear._float32_skin_aabb_index(skin)
    self_check=exact._audit_pair_prepared_first(target,skin,same_surface=True)
    reports=[]
    for key,primitive,p,f in READER["primitives"](mapped,bodies):
        if key==SKIN:
            continue
        chosen, n=clear._candidate_target_rows_for_skin_aabb(f,p,broad)
        ids=np.unique(f[chosen])
        local=np.searchsorted(ids,f[chosen])
        recs=clear._exact_surface_records(p[ids],local)
        recs=[(a,b,c,int(chosen[i]),t) for a,b,c,i,t in recs]
        r=exact._audit_pair_prepared_first(target,recs)
        assert r["aabb_candidate_pairs"]==n
        if r["count"]:
            print(key,r["count"],flush=True)
        reports.append(dict(semantic=key[0],stable_id=key[1],primitive=primitive,target_faces=len(f),**r))
    return self_check,reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack",type=Path)
    parser.add_argument("report",type=Path)
    args=parser.parse_args()
    archive=HERE/"historical-outer-zero-609.npz"
    bone_archive=HERE/"historical-full-bone-native-f32.npz"
    assert hash_file(archive)=="578cee97a3bbecb73c0e1a6d7c26f45ac6cc111a552996f3910866a4a3652a9b"
    assert hash_file(bone_archive)=="e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5"
    receipt=json.loads(args.pack.with_suffix(".receipt.json").read_text())
    assert hash_file(args.pack)==receipt["pack_file_sha256"]
    bodies={row["body_index"]:row for row in receipt["accepted_body_poses"]}
    install_for_float32()
    with np.load(archive,allow_pickle=False) as source:
        xyz,faces=source["points_world_m"],source["faces"]
    with args.pack.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        q,t,registration=fit_bones(mapped,bodies,source_bones(bone_archive))
        aligned=(xyz.astype(float)@q+t).astype("<f4")
        self_check,targets=audit_skin(aligned,faces,mapped,bodies)
    edge=np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1)
    _,n=np.unique(edge,axis=0,return_counts=True)
    r=dict(step=receipt["accepted_step"],pack_sha256=receipt["pack_file_sha256"],
           shape_sha256=hash_file(archive),bone_sha256=hash_file(bone_archive),
           common_frame=registration,surface_banks=len(targets),
           vertices=len(xyz),triangles=len(faces),boundary_edges=int((n==1).sum()),
           non_manifold_edges=int((n>2).sum()),self_intersections=self_check,
           crossings=sum(row["count"] for row in targets),targets=targets,
           admission="offline frozen surface audit only")
    with args.report.open("x") as f:
        json.dump(r,f,indent=2)
        f.write("\n")
    print({k:v for k,v in r.items() if k not in ("targets","common_frame","self_intersections")},
          "registered_bone_error",registration["all_bank_max_face_center_error_m"],
          "self_intersections",self_check["count"],flush=True)


if __name__=="__main__":
    main()
