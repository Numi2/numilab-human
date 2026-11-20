"""Check a candidate's exact native parent/child geometry before compilation."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

import numpy as np

from build_hand_trial import SOURCE, digest
from fit_hand_cavity import PARENT, determinant, table
from fit_hand_interface import coordinates

TOPOLOGY=SOURCE/"NumiHumanElbowExplicitPairedTopology008.hpp"
PARENT_VERTICES=2332
REFINED_VERTICES=2411


def source_array(source,name,columns):
    start=source.index(name+"={{")+len(name)+3
    block=source[start:source.index("\n}};",start)]
    return np.array([int(a) for a in re.findall(r"(\d+)u",block)]).reshape(-1,columns)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("fit",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    ref=coordinates(table(PARENT.with_name(PARENT.name+".nodes.csv")),"rest")
    fit=json.loads(a.fit.read_text())
    child=ref.copy()
    for row in fit["candidate_nodes"]:
        i=row["local_node"]
        assert np.array_equal(ref[i],row["source_world_m"])
        child[i]+=row["world_delta_m"]
    parents=child.reshape(3,REFINED_VERTICES,3)[:,:PARENT_VERTICES].reshape(-1,3)
    source=TOPOLOGY.read_text()
    parent_tets=source_array(source,"kElbowRefinementParentTetrahedra",4)
    child_tets=source_array(source,"kElbowRefinementChildTetrahedra",4)
    owner=source_array(source,"kElbowRefinementChildParentTetrahedra",1).ravel()
    assert parent_tets.shape==(27714,4) and child_tets.shape==(28632,4)
    heavy=parents[parent_tets[owner]]
    matrix=(heavy[:,1:]-heavy[:,:1]).transpose(0,2,1)
    rhs=(child[child_tets]-heavy[:,:1]).transpose(0,2,1)
    bary=np.linalg.solve(matrix,rhs)
    weights=np.concatenate((1.-bary.sum(axis=1,keepdims=True),bary),axis=1)
    violations=(weights.min(axis=(1,2))< -1.e-8)|(weights.max(axis=(1,2))>1.+1.e-8)
    parent_det=determinant(parents,parent_tets)
    child_det=determinant(child,child_tets)
    covered=np.bincount(owner,weights=child_det,minlength=len(parent_det))
    volume_error=np.abs(covered-parent_det)/np.abs(parent_det)
    report={
        "fit_sha256":digest(a.fit),
        "source_topology_sha256":digest(TOPOLOGY),
        "child_point_weight_min":float(weights.min()),
        "child_point_weight_max":float(weights.max()),
        "child_tetrahedra_outside_parent":int(violations.sum()),
        "most_negative_barycentric_child":int(np.unravel_index(weights.argmin(),weights.shape)[0]),
        "max_parent_relative_volume_error":float(volume_error.max()),
        "parent_volume_violations":int(np.count_nonzero(volume_error>1.e-6)),
        "nonpositive_parent_tetrahedra":int(np.count_nonzero(parent_det<=0)),
        "nonpositive_child_tetrahedra":int(np.count_nonzero(child_det<=0)),
    }
    with a.out.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(report))
    assert not (violations.any() or (volume_error>1.e-6).any() or
                (parent_det<=0).any() or (child_det<=0).any()),"explicit parent partition failed"


if __name__=="__main__":
    main()
