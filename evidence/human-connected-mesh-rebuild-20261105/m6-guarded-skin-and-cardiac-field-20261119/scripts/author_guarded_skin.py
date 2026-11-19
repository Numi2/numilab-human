"""Emit a support-pinned source position trial while preserving binding and face rows."""
import csv
import hashlib
import json
import os

import numpy as np

from fit_skin_source import ROOT, PRIOR, decode
from trim_skin_witnesses import positioned_records, exact

def main():
    guard=os.environ.get("NUMI_AUTHOR_BED_FOOTPRINT")=="1"
    input_name="trim-footprint-poses" if guard else "trim-three-poses"
    stem="guarded-footprint" if guard else "guarded-skin"
    certificate=json.loads((ROOT/(input_name+".json")).read_text())
    assert certificate["passed_sampled_geometry_checks"]
    source=ROOT.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin"
    baseline=decode(source.read_bytes())
    delta=np.load(ROOT/(input_name+".npz"))["displacement_m"]
    points=baseline["verts"][:,:3]
    faces=baseline["inds"]
    assert delta.shape==points.shape
    candidate=(points.astype(float)+delta.astype(float)).astype("<f4")
    modified=np.any(points!=candidate,axis=1)
    changed=np.flatnonzero(np.any(modified[faces],axis=1))
    assert len(changed)>0 and np.isfinite(candidate).all()
    with (PRIOR/"wider-release-control-16/native/resting-local-bed-sections.csv").open() as file:
        anchors=np.unique([int(r["source_geometry_index"]) for r in csv.DictReader(file)])
    assert len(anchors)==32 and np.array_equal(candidate[anchors],points[anchors])
    protected=np.union1d(anchors,np.load(ROOT/"bed-section-pins.npz")["vertices"]) if guard else anchors
    assert np.array_equal(candidate[protected],points[protected])
    def area(x):
        t=x[faces].astype(float)
        return np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    from_source,to_source=area(points),area(candidate)
    before=from_source[changed]
    after=to_source[changed]
    ratio=np.einsum("ij,ij->i",before,after)/np.einsum("ij,ij->i",before,before)
    assert min(ratio)>.6
    vertices=baseline["verts"].copy()
    vertices[:,:3]=candidate
    # Adjust the supplied normal by the change of the local area-weighted
    # normal. Keep the old orientation convention and untouched source rows.
    normals=np.zeros_like(candidate,dtype=float)
    for axis in range(3):
        np.add.at(normals,faces[changed,axis],after-before)
    influence=np.unique(faces[changed].ravel())
    influence=np.setdiff1d(influence,protected)
    normal_before=np.zeros_like(normals)
    for axis in range(3):
        np.add.at(normal_before,faces[:,axis],from_source)
    lengths=np.linalg.norm(normal_before[influence],axis=1)
    valid=lengths>0.
    rows=influence[valid]
    normal=vertices[rows,3:6].astype(float)+normals[rows]/lengths[valid,None]
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    assert np.isfinite(normal).all()
    vertices[rows,3:6]=normal.astype("<f4")
    start,end=baseline["vertoff"],baseline["indexoff"]
    payload=baseline["raw"][:start]+vertices.tobytes()+baseline["raw"][end:]
    reached=decode(payload)
    assert np.array_equal(reached["verts"][protected],baseline["verts"][protected])
    rec=positioned_records(candidate,faces,np.arange(len(faces)))
    candidate_tree=exact._prepare_surface_aabb(rec)
    fixed_tree=exact._prepare_surface_aabb(positioned_records(points,faces,np.arange(len(faces))))
    moved=set(changed.tolist())
    def local_pairs(tree):
        records=tree.records
        checked=set()
        pairs=[]
        for a,b in exact._query_surface_aabb((records[i] for i in changed),tree):
            i,j=sorted((a[3],b[3]))
            if i==j or (i,j) in checked:
                continue
            checked.add((i,j))
            pairs.append((records[i],records[j]))
        r=exact._audit_record_pairs(pairs,same_surface=True)
        return {tuple(p) for p in r["triangle_pairs"]}
    original=local_pairs(fixed_tree)
    replacement=local_pairs(candidate_tree)
    added=replacement-original
    output={"source_sha256":hashlib.sha256(baseline["raw"]).hexdigest(),
            "candidate_sha256":hashlib.sha256(payload).hexdigest(),"modified_vertices":int(modified.sum()),
            "support_anchors_byte_equal":len(anchors),"protected_source_rows_byte_equal":len(protected),"minimum_projected_source_area_ratio":float(min(ratio)),
            "source_pairs_touching_edited_faces_before":len(original),
            "source_pairs_touching_edited_faces_after":len(replacement),
            "new_source_pairs":sorted(added)}
    (ROOT/(stem+"-payload.json")).write_text(json.dumps(output,indent=2)+"\n")
    print(output,flush=True)
    if not added:
        (ROOT/(stem+".nhskin")).write_bytes(payload)

if __name__=="__main__":
    main()
