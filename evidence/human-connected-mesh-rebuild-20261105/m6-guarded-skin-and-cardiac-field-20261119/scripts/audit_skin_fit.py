"""Use the owner exact auditing predicates on the fitted Float32 skin positions."""
import json
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"resting-clearance-and-atrium-003"
sys.path.insert(0,str(PRIOR))
from prepare_skin import bindings

def main():
    c,i=bindings.audit.clearance,bindings.audit.intersections
    proposal=np.load(ROOT/"separating-plane-skin-fit.npz")
    paths=(("wider-release-variable-4",2000,"early"),("wider-release-control-16",8000,"late"))
    output=[]
    for run,step,key in paths:
        path=PRIOR/run/f"native/accepted-geometry/step-{step}.mrvpack"
        report=json.loads((PRIOR/f"{run}-exact.json").read_text())
        new=proposal[key]
        receipt=json.loads(path.with_suffix(".receipt.json").read_text())
        assert report["accepted_pack_sha256"]==receipt["pack_file_sha256"]
        previous={(b["semantic"],b["stable_id"]):b for b in report["surfaces"]}
        bodies={v["body_index"]:v for v in receipt["accepted_body_poses"]}
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pack=Path(temp)/"capture.mrvpack"
            with pack.open("wb") as out:
                subprocess.run(["/opt/homebrew/bin/zstd","-dc",str(path)+".zst"],stdout=out,check=True)
            assert bindings.audit.sha(pack)==receipt["pack_file_sha256"]
            old,faces=bindings.read_skin(pack,receipt)
            changed=np.any(new!=old,axis=1)
            changed_faces=np.any(changed[faces],axis=1)
            records=c._exact_surface_records(new,faces)
            query=i._prepare_surface_aabb(records)
            coarse=c._float32_skin_aabb_index(records)
            self_audit=i._audit_pair_prepared_first(query,records,same_surface=True)
            old_self={tuple(x) for x in report["self_intersections"]["triangle_pairs"]}
            new_self={tuple(x) for x in self_audit["triangle_pairs"]}
            skin_self={"before":len(old_self),"after":len(new_self),"removed":len(old_self-new_self),
                       "new":len(new_self-old_self)}
            results=[]
            before_clear=[]; after_clear=[]
            with pack.open("rb") as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as raw:
                for key,_id,v,f in bindings.audit.primitives(raw,bodies):
                    if key==bindings.audit.SKIN:
                        continue
                    selected,broad=c._candidate_target_rows_for_skin_aabb(f,v,coarse)
                    ids=np.unique(f[selected])
                    rr=c._exact_surface_records(v[ids],np.searchsorted(ids,f[selected]))
                    rr=[(a,b,d,int(selected[k]),e) for k,(a,b,d,_,e) in enumerate(rr)]
                    hits=i._audit_pair_prepared_first(query,rr)
                    assert hits["aabb_candidate_pairs"]==broad
                    old_set={tuple(x) for x in previous.pop(key)["triangle_pairs"]}
                    new_set={tuple(x) for x in hits["triangle_pairs"]}
                    assert len(new_set)==hits["count"]
                    assert all(changed_faces[face] for face,_ in old_set^new_set)
                    results.append({"key":list(key),"before":len(old_set),"after":len(new_set),
                                   "new":len(new_set-old_set),"removed":len(old_set-new_set)})
            assert not previous
        totals={k:sum(a[k] for a in results) for k in ("before","after","new","removed")}
        output.append({"time_s":receipt["accepted_time_s"],"pack_sha256":receipt["pack_file_sha256"],
                       "changed_rows":int(changed.sum()),"skin_self":skin_self,
                       "totals":totals,"surfaces":results})
        print(run,skin_self,totals,flush=True)
    data={"resolution":"878 typed banks and skin self intersections",
          "pose_mode":"Frozen capture and Float32 authoring replay",
          "candidate_scale":0.5,"poses":output}
    (ROOT/"separating-plane-skin-audit.json").write_text(json.dumps(data,indent=2)+"\n")

if __name__=="__main__":
    main()
