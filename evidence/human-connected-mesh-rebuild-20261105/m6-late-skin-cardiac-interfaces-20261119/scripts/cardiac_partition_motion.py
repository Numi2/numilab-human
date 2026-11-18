"""Check the exact RA/RV partition for the first rejection and deepest sampled overrun."""
import argparse
import json
from pathlib import Path

import numpy as np
from numilab_human import cardiac_cavity_intersections as inter
from numilab_human import common_atlas_skin_clearance as clear

from inspect_cardiac import TRIAL, parse_payload, sha

def load():
    deck=json.loads((TRIAL/"launch.json").read_text())
    argv=deck["argv"]
    receipt=Path(argv[argv.index("--resting-anatomy-receipt")+1])
    anatomy=Path(argv[argv.index("--torso-anatomy-payload")+1])
    field=json.loads(receipt.read_text())["provenance"]["cardiac_geometry_binding"]["common_field"]
    assert sha(anatomy) == field["anatomy_payload_sha256"]
    file=receipt.parent/field["map"]["path"]
    assert sha(file) == field["map"]["sha256"]
    _, rows = parse_payload(anatomy)
    raw = np.memmap(file,"<f4",mode="r").reshape(-1,7,4)[:,:,:3]
    return rows, raw, field, receipt

def inspect(report, output):
    surfaces,coeff,field,_=load()
    sizes={s["stable_id"]:s for s in field["vertex_ranges"]}
    raw_l,raw_r=[surfaces[k]["vertices6"][:,:3] for k in (318,319)]
    left,right=[surfaces[k]["faces"] for k in (318,319)]
    distinct,identities=np.unique(np.vstack((raw_l,raw_r)),axis=0,return_inverse=True)
    first_ids,second_ids=np.split(identities,[len(raw_l)])
    assert len(np.unique(first_ids))==len(first_ids)
    assert len(np.unique(second_ids))==len(second_ids)
    left=first_ids[left]
    right=second_ids[right]
    maps=np.empty((len(distinct),7,3), dtype="<f4")
    local=lambda k:coeff[sizes[k]["first_vertex"]:sizes[k]["first_vertex"]+sizes[k]["vertex_count"]]
    maps[second_ids]=local(319)
    shared,index_first,_=np.intersect1d(first_ids,second_ids,return_indices=True)
    assert np.array_equal(maps[shared],local(318)[index_first])
    maps[first_ids]=local(318)
    counterpart={tuple(sorted(f)):f for f in right}
    paired=[tuple(sorted(f)) in counterpart for f in left]
    assert sum(paired)==690
    for f in left[paired]:
        r=counterpart[tuple(sorted(f))]
        assert any(np.array_equal(np.roll(r,i),f[::-1]) for i in range(3))
    # Remove just one copy of the proven shared interface from the cross test.
    # Its exact coincident mate stays in the right surface and both self tests.
    exterior_left=left[~np.asarray(paired)]
    out={}
    for label in ("authorized_boundary_q","candidate_q"):
        q=np.asarray(report[label],dtype="<f4").astype(float)
        posed=(distinct+np.einsum("vqd,q->vd",maps,q,dtype=float)).astype("<f4")
        left_rec=clear._exact_surface_records(posed,left)
        right_rec=clear._exact_surface_records(posed,right)
        shared_filtered=[row for row in left_rec if not paired[row[3]]]
        right_tree=inter._prepare_surface_aabb(right_rec)
        ra=inter._audit_pair_prepared_first(inter._prepare_surface_aabb(left_rec),left_rec,same_surface=True)
        rv=inter._audit_pair_prepared_first(right_tree,right_rec,same_surface=True)
        # Shared vertex/edge IDs are authorized only by the exact source quotient,
        # field-byte equality, and the separately identified partition triangles.
        cross=inter._audit_record_pairs(inter._query_surface_aabb(shared_filtered,right_tree),same_surface=True)
        # A record row in one bank can numerically equal a row in another bank.
        # Equal global vertex triples are already covered by the partition check.
        cross["triangle_pairs"]=[list(p) for p in cross["triangle_pairs"]]
        out[label]={"ra":ra,"rv":rv,"right_interface":cross}
        print(label,ra["count"],rv["count"],cross["count"],flush=True)
    result={"method":"Exact Float32 triangle audit in registered heart local frame",
            "shared_partition_faces":sum(paired),"matched_field_vertices":len(shared),
            "input_geometry":report["anatomy_sha256"],"field":report["source_field_sha256"],
            "source_q_report":report,"states":out}
    output.write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("witness",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    inspect(json.loads(args.witness.read_text()),args.output)
