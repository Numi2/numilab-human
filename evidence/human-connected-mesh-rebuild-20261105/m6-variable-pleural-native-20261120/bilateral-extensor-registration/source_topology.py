"""Separate source intersections from posed intersections on unchanged face rows."""
import json
import sys

import numpy as np

from screen import banks, exact, indices, pair_index, records, vertices

def self_pairs(points, faces):
    tree, _, rec = pair_index(points, faces)
    return set(map(tuple,exact._audit_pair_prepared_first(tree,rec,same_surface=True)["triangle_pairs"]))

def topology(faces):
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges.sort(axis=1)
    _, multiplicity=np.unique(edges,axis=0,return_counts=True)
    return {"boundary_edges":int((multiplicity==1).sum()),
            "nonmanifold_edges":int((multiplicity>2).sum())}

if __name__ == '__main__':
    report=[]
    for stable in map(int,sys.argv[2:]):
        _,_,v0,nv,i0,ni,_,_=map(int,records[records[:,6]==stable][0])
        source_face=(indices[i0:i0+ni]-v0).reshape(-1,3)
        used=np.unique(source_face)
        source_face=np.searchsorted(used,source_face)
        source=np.ascontiguousarray(vertices[v0:v0+nv]["position"][used])
        posed,posed_face=banks[51005,stable]
        assert np.array_equal(source_face,posed_face)
        _,original_ids=np.unique(source,axis=0,return_inverse=True)
        _,posed_ids=np.unique(posed,axis=0,return_inverse=True)
        # A diagnostic quotient may join source-coordinate aliases only when
        # those exact coordinates also stay coincident in the captured pose.
        pairs=np.c_[original_ids,posed_ids]
        _,representatives, groups=np.unique(pairs,axis=0,return_inverse=True,return_index=True)
        remap=groups[source_face]
        assert np.array_equal(source[representatives][remap],source[source_face])
        assert np.array_equal(posed[representatives][remap],posed[posed_face])
        raw_original=self_pairs(source,source_face)
        raw_posed=self_pairs(posed,posed_face)
        before=self_pairs(source[representatives],remap)
        after=self_pairs(posed[representatives],remap)
        assert before<=raw_original and after<=raw_posed
        report.append({"stable_id":stable,"source_exact_pairs_no_quotient":len(raw_original),
                       "posed_exact_pairs_no_quotient":len(raw_posed),
                       "coincident_source_and_pose_alias_vertices":len(used)-len(representatives),
                       "source_topology":topology(remap),
                       "source_pairs":sorted(before),"pose_only_pairs":sorted(after-before),
                       "shared_pairs":len(before&after),"pose_pairs":len(after),
                       "positional_only_quotient_diagnostic":True})
    print(json.dumps(report,indent=2))
