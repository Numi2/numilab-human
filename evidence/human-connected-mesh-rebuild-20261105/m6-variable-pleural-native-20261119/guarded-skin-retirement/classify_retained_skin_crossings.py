"""Localize the rest and motion contributions without changing witness policy."""
from collections import Counter
import json

import numpy as np

from fit_connected_skin_offset import HERE, PREVIOUS, PRIOR, SOURCE, decode, hash_file


def banks(path):
    data=json.loads(path.read_text())
    values={}
    for r in data["surfaces"]:
        key=(r["semantic"],r["stable_id"])
        assert key not in values
        values[key]=(r["target_faces"], {tuple(p) for p in r["triangle_pairs"]})
        assert len(values[key][1]) == r["count"]
    return data,values


def main():
    use=[
        HERE.parent/"resting-surface-clearance-001/initial.json",
        PRIOR/"wider-release-variable-4-exact.json",
        HERE/"cardiac-split-native-8000-trimmed-guarded-skin-exact.json",
    ]
    model=decode(SOURCE.read_bytes())
    faces=model["inds"]
    pins=np.load(PREVIOUS/"bed-section-pins.npz")["vertices"]
    fixed=np.zeros(len(model["verts"]),dtype=bool)
    fixed[pins]=True
    frozen=tuple(banks(p) for p in use)
    base=frozen[0][1]
    assert all(t.keys()==base.keys() for _,t in frozen)
    rows=[]
    for r,tables in frozen:
        totals=Counter()
        shared=Counter()
        all_pinned=Counter()
        touched=set()
        details=[]
        for (kind,bank),(count,pairs) in sorted(tables.items()):
            assert count==base[kind,bank][0],(kind,bank)
            if not pairs:
                continue
            involved=np.fromiter((a for a,_ in pairs),dtype=int,count=len(pairs))
            immovable=int(np.count_nonzero(fixed[faces[involved]].all(axis=1)))
            base_overlap=len(pairs & base[kind,bank][1])
            totals[kind]+=len(pairs)
            shared[kind]+=base_overlap
            all_pinned[kind]+=immovable
            touched.update(involved.tolist())
            details.append(dict(semantic=kind,stable_id=bank,pairs=len(pairs),
                                shared_witness_with_initial=base_overlap,
                                all_three_bed_guarded=immovable))
        rows.append(dict(step=r["accepted_step"], pack_sha256=r["accepted_pack_sha256"],
                         unique_skin_faces=len(touched),totals_by_semantic=dict(totals),
                         shared_initial_by_semantic=dict(shared),pinned_by_semantic=dict(all_pinned),
                         banks=sorted(details,key=lambda item:-item["pairs"])))
    out=dict(source_topology_sha256=hash_file(SOURCE),bed_guarded_source_vertices=len(pins),
             exact_audit_sha256=[hash_file(p) for p in use],poses=rows,
             limitation="Shared indices identify repeated triangle-pair witnesses, not a continuous collision test.")
    (HERE/"retained-skin-crossing-inventory.json").write_text(json.dumps(out,indent=2)+"\n")
    for row in rows:
        print({k:v for k,v in row.items() if k!="banks"},flush=True)


if __name__=="__main__":
    main()
