"""Audit the preserved-orientation trial against all retained anatomical banks."""
import json
import mmap
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from solve_skin_enclosure import BASE, END, HERE, ROOT, bindings, decode
from rebind_arm_witnesses import axes

def main():
    factor=.25
    c,i=bindings.audit.clearance,bindings.audit.intersections
    proposal=np.load(HERE/"enclosure-proposal-guarded.npz")
    delta=proposal["normals"]*proposal["delta"][:,None]*factor
    model=decode((HERE/"skin-rebound.nhskin").read_bytes())
    assert np.array_equal(model["verts"][:,:3],proposal["source"])
    baseline=[ROOT/"resting-surface-clearance-001/initial.json",HERE/"rebound-four-exact.json"]
    output=[]
    for path,ref in zip((BASE,END),baseline):
        r=json.loads(path.with_suffix(".receipt.json").read_text())
        report=json.loads(ref.read_text())
        assert report["accepted_pack_sha256"]==r["pack_file_sha256"]
        previous={(b["semantic"],b["stable_id"]):b for b in report["surfaces"]}
        bodies={v["body_index"]:v for v in r["accepted_body_poses"]}
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            pack=Path(temp)/"capture.mrvpack"
            with pack.open("wb") as out:
                subprocess.run(["/opt/homebrew/bin/zstd","-dc",str(path)+".zst"],stdout=out,check=True)
            assert bindings.audit.sha(pack)==r["pack_file_sha256"]
            p,faces=bindings.read_skin(pack,r)
            j=np.einsum("vb,bij->vij",model["weights"],axes(model,r)[0])
            new=(p+np.einsum("vij,vj->vi",j,delta)).astype("<f4")
            records=c._exact_surface_records(new,faces)
            query=i._prepare_surface_aabb(records)
            coarse=c._float32_skin_aabb_index(records)
            results=[]
            with pack.open("rb") as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as raw:
                for key,_id,v,f in bindings.audit.primitives(raw,bodies):
                    if key==bindings.audit.SKIN:
                        continue
                    selected,broad=c._candidate_target_rows_for_skin_aabb(f,v,coarse)
                    ids=np.unique(f[selected])
                    rr=c._exact_surface_records(v[ids],np.searchsorted(ids,f[selected]))
                    rr=[(a,b,d,int(selected[k]),e) for a,b,d,k,e in rr]
                    hits=i._audit_pair_prepared_first(query,rr)
                    assert hits["aabb_candidate_pairs"]==broad
                    old=previous.pop(key)
                    old_set={tuple(x) for x in old["triangle_pairs"]}
                    new_set={tuple(x) for x in hits["triangle_pairs"]}
                    assert len(new_set)==hits["count"] and len(old_set)==old["count"]
                    results.append({"key":list(key),"before":len(old_set),"after":len(new_set),
                                    "new":len(new_set-old_set),"removed":len(old_set-new_set)})
            assert not previous
        totals={k:sum(a[k] for a in results) for k in ("before","after","new","removed")}
        output.append({"time_s":r["accepted_time_s"],"pack_sha256":r["pack_file_sha256"],
                       "totals":totals,"surfaces":results})
        print(r["accepted_time_s"],totals,flush=True)
    data={"resolution":"all typed banks and Float32 world triangle intersections",
          "pose_mode":"Frozen capture perturbed by linearized NHSKIN mapping",
          "candidate_scale":factor,"poses":output}
    (HERE/"enclosure-all-targets.json").write_text(json.dumps(data,indent=2)+"\n")

if __name__=="__main__":
    main()
