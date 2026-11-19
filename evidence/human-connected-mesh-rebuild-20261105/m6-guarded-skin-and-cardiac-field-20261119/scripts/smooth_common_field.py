"""Regularize the same affine basis on both sides of a measured cardiac interface."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from correct_cardiac_field_soft import geometry, ROOT

CORE_M=0.00045
OUTER_M=0.0016


def main():
    # Use the same pre-fit, topology-guarded surface as the prior field trials.
    data_path=ROOT/"atlas-training-original.bin"
    raw,rows,faces,windows=geometry(data_path)
    ref=rows[:,0].astype(float)
    with (ROOT/"atlas-training-held-witnesses.csv").open() as stream:
        face_pairs={(int(row["first_face"]),int(row["second_face"]))
                    for row in csv.DictReader(stream)}
    with (ROOT.parent/"resting-clearance-and-atrium-003/common-cardiac-tight-witnesses.csv").open() as stream:
        face_pairs.update((int(row["first_face"]),int(row["second_face"]))
                          for row in csv.DictReader(stream))
    witness_faces=np.unique(np.asarray(sorted(face_pairs)))
    witness_rows=np.unique(faces[witness_faces])
    seeds=ref[witness_rows]
    near=cKDTree(seeds).query_pairs(2.05*OUTER_M,output_type="ndarray")
    graph=coo_matrix((np.ones(len(near)),(near[:,0],near[:,1])),shape=(len(seeds),len(seeds)))
    count,clusters=connected_components(graph,directed=False)
    vertices=cKDTree(ref)
    output=rows.copy()
    modified=np.zeros(len(rows),dtype=bool)
    receipt=[]
    for group in range(count):
        center=seeds[clusters==group]
        local=np.unique(np.concatenate(vertices.query_ball_point(center,OUTER_M)))
        assert not modified[local].any()
        dist=cKDTree(center).query(ref[local])[0]
        core=dist<=CORE_M
        # Fit the neighborhood around the feature, with negligible weight in
        # the measured sliver region so the discontinuity does not drive it.
        fit=dist>.9*CORE_M
        average=ref[local[fit]].mean(axis=0)
        design=np.c_[np.ones(len(local)),(ref[local]-average)/OUTER_M]
        b=rows[local,1:].astype(float).reshape(len(local),21)
        affine,_,rank,_=np.linalg.lstsq(design[fit],b[fit],rcond=1.e-8)
        assert rank==4
        new=design@affine
        shape=np.clip((OUTER_M-dist)/(OUTER_M-CORE_M),0.,1.)
        blends=(shape**3*(10+shape*(-15+6*shape)))[:,None]
        d=blends*(new-b)
        output[local,1:]=(b+d).reshape(-1,7,3).astype("<f4")
        modified[local]=True
        receipt.append({"seed_rows":int((clusters==group).sum()),"affected_rows":len(local),
                        "common_affine_core_rows":int(core.sum()),
                        "max_coefficient_component_delta_m":float(abs(d).max())})
    assert np.array_equal(output[:,0],rows[:,0])
    dest=ROOT/"smooth-common-probe.bin"
    dest.write_bytes(raw[:12]+output.tobytes()+raw[12+rows.nbytes:])
    report={"changed_rows":int(np.any(output!=rows,axis=(1,2)).sum()),"retained_vertices":len(rows),
            "clusters":receipt,"core_radius_m":CORE_M,"outer_radius_m":OUTER_M,
            "inputs_sha256":hashlib.sha256(raw).hexdigest(),"probe_windows":len(windows),
            "default_model_changed":False}
    (ROOT/"smooth-common-probe.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)

if __name__=="__main__":
    main()
