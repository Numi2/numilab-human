#!/usr/bin/env python3
import hashlib, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
CAND=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/candidate.pkl"
CAND_SHA="8607d93a055a2045e823f80b4e04b6d6dcd6f5d8bfa95b442485a35b555918b1"
ATTEMPT142=ROOT/"attempt-142-right-current8000-witness-edge-midpoint-refinement-001.py"
BUNDLE=A/"../.."/"numi-human-fds132-m6-1350"/"attempt-156-right-candidate150-carrier-assets-001"
OUT=ROOT/"attempt-157-right-candidate150-late151999-f64-critical-001"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise RuntimeError(m)
def main():
    require(not OUT.exists(),"preserve prior attempt157 output")
    OUT.mkdir()
    require(sha(CAND)==CAND_SHA,"candidate150 hash changed")
    spec=importlib.util.spec_from_file_location("attempt142_helper",ATTEMPT142)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    loaded=m.loadref()
    _,_,core,_,_,_,_,_,_,ci,bind,default_pose,poses,contexts,input_hashes,helper=loaded
    require("step151999" in poses and "step151999" in contexts,"pinned late151999 pose is unavailable")
    pose=poses["step151999"];pose_meta=contexts["step151999"]
    require(pose_meta["step"]==151999 and Path(pose_meta["pack_path"]).name=="step-151999.mrvpack","late pose identity mismatch")
    cp=pickle.loads(CAND.read_bytes())
    require(list(map(int,cp["core_body_indices"]))==list(map(int,core)),"candidate/helper 21-core order mismatch")
    rows={};gpos={};weights={}
    for sid in (131,133):
        sf=cp["surfaces"][sid]
        ids=list(map(int,sf["global_vertex_ids"]))
        points=np.asarray(sf["vertices_float32_m"],dtype=np.float32)
        ww=np.asarray(sf["core_weights_float64"],dtype=np.float64)
        faces=np.asarray(sf["faces"],dtype=np.int64)
        fg=np.asarray(sf["faces_global"],dtype=np.int64)
        require(points.shape==(len(ids),3) and ww.shape==(len(ids),21),f"stable{sid} point/field shape mismatch")
        require(np.array_equal(np.asarray(ids,dtype=np.uint64)[faces],fg),f"stable{sid} local/global face mapping mismatch")
        require(np.all(ww>0) and np.isfinite(ww).all() and np.max(np.abs(ww.sum(axis=1)-1.0))<1e-12,f"stable{sid} full21 partition invalid")
        for gid,p,w in zip(ids,points,ww):
            if gid in gpos:
                require(np.array_equal(gpos[gid],p) and np.array_equal(weights[gid],w),f"shared interface data mismatch gid={gid}")
            else:
                gpos[gid]=p.copy();weights[gid]=w.copy()
        rows[sid]={"ids":ids,"points":points,"faces":faces,"fg":fg}
    progress={"schema":"numi.human.right-candidate150-late151999-f64-critical-157.progress.v1","status":"running_bounded_late151999_only","candidate_sha256":CAND_SHA,"pose":pose_meta,"scope":"Only stable131/133 and their joined external boundary at retained P011 baseline step151999; CPU Float64 binding/weight accumulation and exact predicates. No source/current retest or GPU work.","completed":{}}
    progress_path=OUT/"progress.json";progress_path.write_text(json.dumps(progress,sort_keys=True,indent=2)+"\n")
    pos={gid:m.world(gpos[gid].astype(np.float64),weights[gid],core,bind,pose) for gid in gpos}
    row_results={}
    for sid in (131,133):
        points=np.asarray([pos[gid] for gid in rows[sid]["ids"]],dtype=np.float64)
        result=m.exactaudit(points,rows[sid]["faces"],ci,f"late151999-f64-{sid}")
        row_results[str(sid)]=result
        progress["completed"][str(sid)]=result
        progress_path.write_text(json.dumps(progress,sort_keys=True,indent=2)+"\n")
    union=m.unionaudit(rows,pos,ci,"late151999-f64-union")
    critical={"schema":"numi.human.right-candidate150-late151999-f64-critical-157.v1","status":"complete","candidate_path":str(CAND),"candidate_sha256":CAND_SHA,"candidate_bundle_path":str(BUNDLE),"pose":pose_meta,"pose_pack_sha256":pose_meta["pack_sha256"],"pose_receipt_sha256":pose_meta["receipt_sha256"],"core_body_indices":list(map(int,core)),"arithmetic":"Existing attempt142 source/TISS binding helper world() with Float64 point, binding, body-pose, weight accumulation; exact Fraction predicates on resulting binary64 coordinates. CPU only; not native GPU.","scope":"Candidate150 retained baseline late151999 step only; source/current8000 gates are separate pinned reports.","rows":row_results,"union":union,"input_sha256":{str(ATTEMPT142):sha(ATTEMPT142),str(CAND):sha(CAND),**input_hashes}}
    report=OUT/"report.json";report.write_text(json.dumps(critical,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"complete","report":str(report),"report_sha256":sha(report),"candidate_sha256":CAND_SHA,"pose_pack_sha256":pose_meta["pack_sha256"],"pose_receipt_sha256":pose_meta["receipt_sha256"],"rows":{s:{"self_count":v["self_count"],"components":v["components"],"faces":v["faces"],"aabb_candidates":v["aabb_candidates"]} for s,v in row_results.items()},"union":{"self_count":union["self_count"],"components":union["components"],"faces":union["faces"],"aabb_candidates":union["aabb_candidates"]}},sort_keys=True))
if __name__=="__main__":main()
