"""Compare the matched native terminal states and measure protected witnesses."""
from collections import Counter
import json

import numpy as np

from fit_clearance_patches import HERE, decode, mesh_edges, clear
from measure_crossing_curves import audit


def load(path):
    return json.loads(path.read_text())


def report(path, faces, protected):
    raw=load(path)
    pairs={}
    fixed=Counter()
    for b in raw["surfaces"]:
        key=b["semantic"],b["stable_id"]
        pairs[key]=set(map(tuple,b["triangle_pairs"]))
        fixed[key[0]]+=sum(bool(protected[faces[i]].all()) for i,_ in pairs[key])
    self_pairs=set(map(tuple,raw["self_intersections"]["triangle_pairs"]))
    return pairs,self_pairs,dict(accepted_step=raw["accepted_step"],pack_sha256=raw["accepted_pack_sha256"],
        tissue_pairs=sum(map(len,pairs.values())),self_pairs=len(self_pairs),
        fully_protected_skin_face_pairs=sum(fixed.values()),
        fully_protected_pairs_by_typed_semantic=dict(fixed))


def main():
    old=HERE/"connected-skin-native"
    new=HERE/"source-clearance-patches-native"
    exact_old=old/"skin-exact.json"
    exact_new=new/"step-8000-skin-exact.json"
    data=decode((HERE/"source-clearance-patches.nhskin").read_bytes())
    faces=data["inds"]
    e,n=mesh_edges(faces)
    protected=np.zeros(len(data["verts"]),dtype=bool)
    protected[e[n!=2]]=True
    protected[clear._exact_coordinate_tied_source_vertex_ids(data["verts"][:,:3])]=True
    protected[np.load(HERE.parent/"resting-clearance-and-atrium-004/bed-section-pins.npz")["vertices"]]=True
    assert protected.sum()==9588
    before,s_before,a=report(exact_old,faces,protected)
    after,s_after,b=report(exact_new,faces,protected)
    _,_,early=report(new/"step-2015-skin-exact.json",faces,protected)
    assert before.keys()==after.keys()
    obj=dict(control=a,candidate=b,earlier_candidate_capture=early,
        introduced_tissue_pairs=sum(len(after[k]-v) for k,v in before.items()),
        removed_tissue_pairs=sum(len(v-after[k]) for k,v in before.items()),
        introduced_self_pairs=len(s_after-s_before),
        removed_self_pairs=len(s_before-s_after),
        cleared_typed_banks=[k for k in sorted(before) if before[k] and not after[k]],
        bank_differences=[dict(semantic=k[0],stable_id=k[1],before=len(v),after=len(after[k]),
                              removed=len(v-after[k]),added=len(after[k]-v))
                          for k,v in sorted(before.items()) if v!=after[k]])
    assert not obj["introduced_self_pairs"] and not obj["introduced_tissue_pairs"],obj
    curves_control=load(HERE/"native-intersection-curves.json")["candidate"]
    assert curves_control["pack_sha256"]==a["pack_sha256"]
    curves_candidate=audit(new,exact_new)
    for target,source in ((a,curves_control),(b,curves_candidate)):
        target.update(tissue_curve_length_m=source["tissue_curve_length_m"],
                      tissue_curve_components=sum(x["connected_curve_components"] for x in source["tissue"].values()),
                      self_curve_length_m=source["self_crossing"]["length_m"],
                      self_curve_components=source["self_crossing"]["connected_curve_components"])
    obj["sample_limit"]="Exact captured Float32 surfaces. Curve length is an intersection extent. Fully protected rows are fixed during this source-only repair."
    (new/"geometry-comparison.json").write_text(json.dumps(obj,indent=2)+"\n")
    print(json.dumps(obj,indent=2),flush=True)


if __name__=="__main__":
    main()
