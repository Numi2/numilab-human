"""Recover matched link transforms before reusing the archived outer skin."""
import argparse
from collections import defaultdict
import json
import mmap
from pathlib import Path
import runpy

import numpy as np
from scipy.spatial import cKDTree

from fit_cardiac_volume_grid import HERE, hash_file

READER=runpy.run_path(str(HERE.parent/"resting-surface-clearance-001/audit.py"))
BONE=51004


def rigid_fit(x,y):
    xc=x.mean(axis=0)
    yc=y.mean(axis=0)
    u, _s, vt=np.linalg.svd((x-xc).T@(y-yc))
    if np.linalg.det(u@vt)<0:
        u[:,-1]*=-1
    q=u@vt
    t=yc-xc@q
    return q,t,np.linalg.norm(x@q+t-y,axis=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_manifest",type=Path)
    ap.add_argument("current_pack",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    old=json.loads(args.source_manifest.read_text())["operands"]
    by={i["bone"]:i for i in old}
    assert len(by)==185
    receipt=json.loads(args.current_pack.with_suffix(".receipt.json").read_text())
    assert hash_file(args.current_pack)==receipt["pack_file_sha256"]
    bodies={r["body_index"]:r for r in receipt["accepted_body_poses"]}
    rows=[]
    support=defaultdict(list)
    missing=[]
    with args.current_pack.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        for (kind,number),_,points,tri in READER["primitives"](mapped,bodies):
            if kind!=BONE:
                continue
            spec=by.pop(number)
            assert spec["face_count"]==len(tri)
            file=Path(spec["operand_path"])
            if not file.is_file():
                missing.append(dict(stable_id=number, old_body_index=spec["body"]))
                continue
            assert hash_file(file)==spec["operand_sha256"]
            with np.load(file) as z:
                points_old=z["points_world_m_f64"]
                old_to_order=np.argsort(z["bank_stable_face_row"])
                assert np.array_equal(z["bank_stable_face_row"][old_to_order],np.arange(len(tri)))
                original_tri=z["faces_f64"]
                original_order=np.argsort(z["face_id"])
                hypotheses={"stable_face_row":original_tri[old_to_order]}
                if np.array_equal(z["face_id"][original_order], np.arange(len(tri))):
                    hypotheses["source_face_id"]=original_tri[original_order]
            y=points[tri].astype(float).mean(axis=1)
            fits=[]
            for row_order, old_faces in hypotheses.items():
                centers=points_old[old_faces].mean(axis=1)
                rotation,translation,error=rigid_fit(centers,y)
                fits.append((float(error.max()), row_order, centers, rotation, translation, error))
            _, row_order, x, q, t, face_error=min(fits, key=lambda item:item[0])
            dm,_=cKDTree(points).query(points_old@q+t)
            reciprocal,_=cKDTree(points_old@q+t).query(points)
            valid=max(float(dm.max()),float(reciprocal.max()),float(face_error.max()))<=5.e-6
            row=dict(stable_id=number, old_body_index=spec["body"],faces=len(x), row_order=row_order,
                     face_center_max_m=float(face_error.max()),
                     forward_max_m=float(dm.max()),reverse_max_m=float(reciprocal.max()),
                     strict_correspondence=bool(valid), source_sha256=spec["operand_sha256"])
            rows.append(row)
            if valid:
                support[spec["body"]].append((x,y))
    assert not by
    converted=[]
    for body,components in sorted(support.items()):
        x,y=map(np.concatenate,zip(*components))
        q,t,e=rigid_fit(x,y)
        converted.append(dict(body=body, bank_count=len(components),rotation_row=q.tolist(),
                              translation_m=t.tolist(), max_registered_center_error_m=float(e.max()),
                              strict_correspondence=bool(e.max()<=5.e-6)))
    output=dict(old_manifest_sha256=hash_file(args.source_manifest), current_pack_sha256=receipt["pack_file_sha256"],
                step=receipt["accepted_step"],banks=rows,missing=missing,links=converted,
                limit="Topology stable-ID and geometry registration witness only. No skin transfer is implied.")
    with args.output.open("x") as f:
        json.dump(output,f,indent=2)
        f.write("\n")
    print(dict(comparable=len(rows),registered=sum(x["strict_correspondence"] for x in rows),
               registered_links=sum(x["strict_correspondence"] for x in converted),
               missing_banks=len(missing),maximum_registered_m=max(
                   (x["max_registered_center_error_m"] for x in converted if x["strict_correspondence"]),default=None)),flush=True)


if __name__=="__main__":
    main()
