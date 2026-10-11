#!/usr/bin/env python3
"""Refresh an accepted offline NHSKIN candidate against the exact 30-surface capture cohort.

This is a bounded preparation/export operation. It never launches native simulation.
"""
from __future__ import annotations
import copy, gzip, hashlib, importlib.util, json, os, struct, sys, time, traceback
from pathlib import Path
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
OUT=A/"skin-candidate-30-pose-preparation-001/result-005"
SCRIPT=Path(__file__).resolve()
FIT=A/"stable152-skin-fit-001/result-014-resume-004"
FIT_REPORT=FIT/"candidate-report.json"
FIT_NPY=FIT/"candidate-source-positions-f32.npy"
FIT_EVENTS=FIT/"attempts.jsonl"
FIT_TARGETS=FIT/"attempts/attempt-0002-target-audits.json.gz"
FIT_PREFLIGHT=FIT/"preflight.json"
BASE_DIR=R/"skin-resting-multipose-clearance-1218/package-preparation-005/composed-candidate-001"
BASE_SKIN=BASE_DIR/"bodyparts3d-myosim-skinned-shell.nhskin"
BASE_SKIN_MAN=BASE_DIR/"common-atlas-skin-geometry-registration.manifest.json"
BASE_SCENE=BASE_DIR/"resting-supine-scene.manifest.json"
CURRENT=A/"thirty-surface-native-composition-001"
CURRENT_RECEIPT=CURRENT/"resting-anatomy-receipt.json"
CURRENT_NHTISS=CURRENT/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CURRENT_NHTISS_MAN=CURRENT/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
CURRENT_COMPOSITION=CURRENT/"composition-result.json"
CURRENT_NATIVE_REPORT=CURRENT/"native-verification-001/report.json"
CURRENT_RUN_DECL=CURRENT/"baseline/run-declaration.json"
OLD_CAPTURE=A/"twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry"
NEW_CAPTURE=CURRENT/"baseline/native-run/accepted-geometry"
STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
FORWARD=Path("/Users/n/numi-human-resting-evidence-20261005/native-common-skin-multipose-forward-model-915.py")
TISS_FORWARD=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
TISS_SOURCE=R/"passive-neck-back-coverage-1281/candidate-005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS_SOURCE_MAN=TISS_SOURCE.with_suffix(".manifest.json")
RESP=Path("/Users/n/numi-human-resting-resp-source-20261006")
CA=Path("/Users/n/numi-human-self-audit-resume-1265/src/numilab_human/common_atlas_skin_clearance.py")
CI=CA.with_name("cardiac_cavity_intersections.py")
HUMAN_SOURCE=CA.parent.parent
GEOM=HUMAN_SOURCE/"numilab_human/common_atlas_skin_geometry_registration.py"
PREFLIGHT=HUMAN_SOURCE/"numilab_human/skin_source_payload_preflight.py"
ANATOMY=HUMAN_SOURCE/"numilab_human/resting_anatomy.py"
HUMAN_RUNTIME=Path("/Users/n/numi-human-positive-winding-1279")
REGISTRATION=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json")
SKIN_KEY=(51007,1)
OFFLINE_KEYS={(51005,151),(51005,152)}
REFRESH_KEYS={(51005,37),(51005,38),(51005,39),(51005,40),(51005,67),(51005,68),(51005,75),(51005,76)}

def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha(p:Path)->str:
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def pin(p:Path)->dict:
    p=Path(p).resolve()
    if not p.is_file():raise FileNotFoundError(str(p))
    return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def write_json(p:Path,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
def f32sha(x):return sha_bytes(np.ascontiguousarray(x,dtype="<f4").tobytes())
def pose_map(rec,key):
    out={}
    for item in rec.get(key,[]):
        i=int(item["body_index"])
        if i in out:raise ValueError(f"duplicate body pose {i} in {key}")
        out[i]=(np.asarray(item["position_m"],dtype="<f4"),np.asarray(item["quaternion_xyzw"],dtype="<f4"))
    return out
def load_skin(path):
    raw=Path(path).read_bytes(); magic,abi,nb,nv,ni,fp,arch=struct.unpack_from("<8s5I32s",raw)
    vo=60+36*nb;io=vo+56*nv;wo=io+4*ni
    need(magic==b"NHSKIN1\0" and abi==5 and len(raw)==wo+4*nv*nb,"unsupported NHSKIN layout")
    vr=np.frombuffer(raw,"<f4",14*nv,vo).reshape(nv,14)
    faces=np.frombuffer(raw,"<u4",ni,io).reshape(-1,3).astype(np.int64)
    bind_u=np.frombuffer(raw,"<u4",9*nb,60).reshape(nb,9).copy()
    bind=np.frombuffer(raw,"<f4",9*nb,60).reshape(nb,9).copy()
    return raw,{"nb":int(nb),"nv":int(nv),"ni":int(ni),"fp":int(fp),"pos":vr[:,:3].copy(),"normals":vr[:,3:6].copy(),"bind_u":bind_u,"bind":bind,"faces":faces,"weights":np.frombuffer(raw,"<f4",nv*nb,wo).reshape(nv,nb).copy(),"vertex_offset":vo,"vertex_end":io,"weight_offset":wo}
def load_tissue(path):
    raw=Path(path).read_bytes();magic,abi,nr,nb,nv,ni,fp,src=struct.unpack_from("<8s6I32s",raw);bo=64+32*nr;vo=bo+36*nb;io=vo+56*nv
    need(magic==b"NHTISS4\0" and abi==5 and len(raw)==io+4*ni,"unsupported NHTISS layout")
    rec=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(-1,8)
    bind=np.frombuffer(raw,dtype=np.dtype([("core","<u4"),("value","<f4",(8,))]),count=nb,offset=bo)
    ind=np.frombuffer(raw,"<u4",count=ni,offset=io)
    return {"path":Path(path),"raw":raw,"records":rec,"bindings":bind,"indices":ind,"bo":bo,"vo":vo}
def tissue_row(t,sid):
    hits=[r for r in t["records"] if int(r[6])==sid];need(len(hits)==1,f"stable {sid} row count {len(hits)}")
    r=hits[0];fb,bc,fv,vc,fi,ic=map(int,r[:6]);raw=t["raw"]
    pos=np.ndarray((vc,3),dtype="<f4",buffer=raw,offset=t["vo"]+fv*56,strides=(56,4)).copy()
    local=np.ndarray((vc,4),dtype="<u4",buffer=raw,offset=t["vo"]+fv*56+24,strides=(56,4)).copy()
    weights=np.ndarray((vc,4),dtype="<f4",buffer=raw,offset=t["vo"]+fv*56+40,strides=(56,4)).copy()
    faces=t["indices"][fi:fi+ic].reshape(-1,3).astype(np.int64)-fv
    return {"sid":sid,"fb":fb,"bc":bc,"fv":fv,"vc":vc,"fi":fi,"ic":ic,"positions":pos,"local":local,"weights":weights,"faces":faces}
def rotate32(q,p):
    f=np.float32;q=np.asarray(q,dtype=f);p=np.asarray(p,dtype=f);a=q[:3]
    cr=np.asarray([f(f(a[1]*p[2])-f(a[2]*p[1])),f(f(a[2]*p[0])-f(a[0]*p[2])),f(f(a[0]*p[1])-f(a[1]*p[0]))],dtype=f)
    tw=np.asarray([f(f(2)*v) for v in cr],dtype=f)
    co=np.asarray([f(f(f(q[3]*tw[0])+f(a[1]*tw[2]))-f(a[2]*tw[1])),f(f(f(q[3]*tw[1])+f(a[2]*tw[0]))-f(a[0]*tw[2])),f(f(f(q[3]*tw[2])+f(a[0]*tw[1]))-f(a[1]*tw[0]))],dtype=f)
    return np.asarray([f(p[i]+co[i]) for i in range(3)],dtype=f)
def tissue_forward(row,t,poses):
    f=np.float32;out=np.zeros_like(row["positions"],dtype=f)
    for vi in range(len(out)):
        accum=np.zeros(3,dtype=f)
        for slot in range(4):
            w=f(row["weights"][vi,slot])
            if w<=f(0):continue
            li=int(row["local"][vi,slot]);need(li<row["bc"],"offline target local binding index out of range")
            b=t["bindings"][row["fb"]+li];core=int(b["core"]);need(core in poses,f"offline target missing body pose {core}")
            val=b["value"];rr=rotate32(val[3:7],row["positions"][vi])
            loc=np.asarray([f(val[j]+f(f(rr[j])*val[7])) for j in range(3)],dtype=f)
            bp,bq=poses[core];wr=rotate32(bq,loc);world=np.asarray([f(bp[j]+wr[j]) for j in range(3)],dtype=f)
            accum=np.asarray([f(accum[j]+f(w*world[j])) for j in range(3)],dtype=f)
        out[vi]=accum
    return out
def row_sig(xyz,surface):
    faces=np.asarray(surface["faces"],dtype=np.int64)
    need(faces.ndim==2 and faces.shape[1]==3 and faces.size>0,"empty/malformed target surface")
    ids=np.unique(faces); local=np.searchsorted(ids,faces).astype("<i8")
    points=np.asarray(xyz[ids],dtype="<f4")
    h=hashlib.sha256(); h.update(struct.pack("<QQQ",int(surface["body"]),len(ids),len(faces)));h.update(local.tobytes());h.update(points.tobytes())
    return {"sha256":h.hexdigest(),"body":int(surface["body"]),"vertex_count":len(ids),"face_count":len(faces),"face_identity_sha256":sha_bytes(local.tobytes()),"position_f32_sha256":sha_bytes(points.tobytes())}
def main():
    if OUT.exists():raise RuntimeError("refuse existing result directory "+str(OUT))
    OUT.mkdir(parents=True);(OUT/"progress.jsonl").touch()
    t0=time.monotonic()
    def log(event,**values):
        row={"event":event,"elapsed_s":time.monotonic()-t0,**values}
        with (OUT/"progress.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
        print(json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
    try:
        sys.path.insert(0,str(CA.parent.parent));sys.path.insert(0,str(HUMAN_SOURCE))
        from numilab_human import common_atlas_skin_clearance as ca
        from numilab_human import common_atlas_skin_geometry_registration as geom
        from numilab_human import skin_source_payload_preflight as pre
        from numilab_human import resting_anatomy as anatomy
        need(Path(ca.__file__).resolve()==CA.resolve(),"unexpected clearance owner import")
        need(Path(geom.__file__).resolve()==GEOM.resolve(),"unexpected skin composition owner import")
        need(Path(pre.__file__).resolve()==PREFLIGHT.resolve(),"unexpected NHSKIN decoder import")
        need(Path(anatomy.__file__).resolve()==ANATOMY.resolve(),"unexpected anatomy receipt owner import")
        skin_raw,skin=load_skin(BASE_SKIN); fit=np.load(FIT_NPY,allow_pickle=False)
        fit_report=json.loads(FIT_REPORT.read_text());fit_events=[json.loads(x) for x in FIT_EVENTS.read_text().splitlines()]
        fit_event=next(e for e in reversed(fit_events) if e.get("status")=="accepted")
        need(fit_report.get("status")=="inferred_engineering_clearance_candidate_pending_native_replay","fit candidate status changed")
        need(fit.shape==(skin["nv"],3) and fit.dtype==np.dtype("<f4") and np.isfinite(fit).all(),"fit NPY layout/nonfinite")
        need(f32sha(fit)==fit_report["candidate_source_positions_f32_sha256"],"fit full-source F32 identity differs from report")
        need(sha(FIT_NPY)==fit_report["candidate_source_positions_npy_sha256"],"fit NPY container identity differs")
        ref=np.unique(skin["faces"]);compact=np.searchsorted(ref,skin["faces"])
        need(f32sha(fit[ref])==fit_event["source_positions_f32_sha256"],"compact source identity differs from accepted event")
        need(len(fit_event["self_audits_by_pose"])==8 and all(r.get("count")==0 and not r.get("degenerate_face_rows") for r in fit_event["self_audits_by_pose"]),"accepted fit candidate lacks zero exact self proof")
        need(all(r.get("face_index_sha256")==ca._face_index_sha256(compact) for r in fit_event["self_audits_by_pose"]),"fit self proof topology identity differs")
        need(sha(FIT_TARGETS)==fit_event["target_audits_sha256"],"accepted fit target container pin differs")
        with gzip.open(FIT_TARGETS,"rt",encoding="utf-8") as f: old_tables=json.load(f)
        need(isinstance(old_tables,list) and len(old_tables)==8,"accepted candidate tables do not cover 8 poses")
        legacy,nonocular,_,_=ca._load_target_inventory(INV);allkeys=set(legacy)|OFFLINE_KEYS
        expected_text={f"{s}:{i}" for s,i in allkeys};refresh_text={f"{s}:{i}" for s,i in REFRESH_KEYS}
        need(len(legacy)==859 and len(allkeys)==861 and REFRESH_KEYS.issubset(legacy),"target inventory changed")
        for i,t in enumerate(old_tables):
            need(set(t)==expected_text and len(t)==861,f"accepted table coverage mismatch at pose {i}")
            need(all(type(v.get("count")) is int and v["count"]==len(v.get("triangle_pairs",[])) and not v.get("degenerate_face_rows") for v in t.values()),f"accepted table malformed at pose {i}")
        fit_table_hashes=[ca._baseline_target_pair_table_sha256(t) for t in old_tables]
        need(all(sum(v["count"] for v in t.values())==0 for t in old_tables),"accepted fit candidate target tables are not all zero")

        # Read paired capture receipts and verify the old/current cohorts represent the same accepted states.
        old_receipts=[];new_receipts=[];old_poses=[];new_poses=[]
        oldpacks=[];newpacks=[];captured_old=[];captured_new=[]
        for step in STEPS:
            op=OLD_CAPTURE/f"step-{step}.mrvpack";orcp=OLD_CAPTURE/f"step-{step}.receipt.json"
            npack=NEW_CAPTURE/f"step-{step}.mrvpack";nrcp=NEW_CAPTURE/f"step-{step}.receipt.json"
            orec=json.loads(orcp.read_text());nrec=json.loads(nrcp.read_text())
            need(orec.get("accepted_step")==step and nrec.get("accepted_step")==step,"capture accepted-step mismatch")
            need(orec.get("pack_file_sha256")==sha(op) and nrec.get("pack_file_sha256")==sha(npack),"capture receipt/pack hash mismatch")
            need(orec.get("accepted_registered_body_poses")==nrec.get("accepted_registered_body_poses"),f"registered body poses differ old/current at {step}")
            need(orec.get("accepted_body_poses")==nrec.get("accepted_body_poses"),f"body pose records differ old/current at {step}")
            need(orec.get("accepted_respiratory_motion")==nrec.get("accepted_respiratory_motion"),f"respiratory state differs old/current at {step}")
            need(orec.get("initial_anatomical_registration")==nrec.get("initial_anatomical_registration"),f"initial registration differs old/current at {step}")
            old_receipts.append(orec);new_receipts.append(nrec);oldpacks.append(op);newpacks.append(npack)
            old_poses.append(pose_map(orec,"accepted_body_poses"));new_poses.append(pose_map(nrec,"accepted_body_poses"))
            need(len(old_poses[-1])==157 and len(new_poses[-1])==157,"accepted body pose inventory is incomplete")
        log("capture_state_identity_passed",steps=STEPS,body_pose_count=157,old_new_registered_poses_equal=True,respiratory_states_equal=True)

        # Recreate the exact 915 skin source-to-world map and validate its baseline bit replay.
        initial=new_receipts[0]
        for step,pack in zip(STEPS,newpacks):
            xyz,surfs,meta=ca._pack_surfaces(pack,legacy|{SKIN_KEY})
            need(SKIN_KEY in surfs,"current30 pack lacks NHSKIN key")
            sf=np.asarray(surfs[SKIN_KEY]["faces"],dtype=np.int64);base=int(sf.min())
            need(len(sf)==len(skin["faces"]) and np.array_equal(sf-base,skin["faces"]),f"skin topology differs in current30 pack at {step}")
            cw=np.asarray(xyz[base:base+skin["nv"]],dtype="<f4")[ref].copy()
            captured_new.append(cw)
            # Old and new packs can differ in packed offsets; compare extracted skin world positions exactly.
            ox,osurf,_=ca._pack_surfaces(OLD_CAPTURE/f"step-{step}.mrvpack",{SKIN_KEY})
            osf=np.asarray(osurf[SKIN_KEY]["faces"],dtype=np.int64);obase=int(osf.min())
            ow=np.asarray(ox[obase:obase+skin["nv"]],dtype="<f4")[ref]
            need(np.array_equal(ow,cw),f"captured skin world Float32 differs old/current at {step}")
            captured_old.append(ow.copy())
            del xyz,surfs,meta,ox,osurf
        captured=np.stack(captured_new).astype("<f4")
        fs=importlib.util.spec_from_file_location("skin_forward_915_current30",FORWARD);fm=importlib.util.module_from_spec(fs);fs.loader.exec_module(fm)
        need(Path(fm.__file__).resolve()==FORWARD.resolve(),"unexpected 915 forward module")
        states=[{"step":s,"body_poses":r["accepted_registered_body_poses"],"respiratory_motion":r["accepted_respiratory_motion"]} for s,r in zip(STEPS,new_receipts)]
        forward,map_report,selector_ids=fm.build_forward(skin=skin,source_positions=skin["pos"],referenced_ids=ref,captured_by_pose=captured.astype(float),state_receipts=states,
             initial_body_poses=initial["initial_anatomical_registration"]["body_poses"],map_receipt=initial,
             anatomy_parameters_path=initial["skin_source_mapping"]["anatomy_parameters"]["path"],respiration_source=RESP,clearance_module=ca)
        base_result=forward(skin["pos"].astype("<f4"));need(base_result.get("diagnostics",{}).get("admissible") is True,"915 current30 baseline forward inadmissible")
        base_world=np.asarray(base_result["world_positions_by_pose"],dtype="<f4")
        need(np.array_equal(base_world,captured),"915 source baseline does not bit-replay all current30 captured skin worlds")
        cand_result=forward(fit);need(cand_result.get("diagnostics",{}).get("admissible") is True,"915 fitted candidate forward inadmissible")
        candidate_world=np.asarray(cand_result["world_positions_by_pose"],dtype="<f4")
        need(candidate_world.shape==captured.shape and np.isfinite(candidate_world).all(),"candidate current30 world layout/nonfinite")
        world_hashes=[ca._float32_xyz_sha256(x) for x in candidate_world]
        fit_world_hashes=[r["candidate_world_f32_sha256"] for r in fit_event["self_audits_by_pose"]]
        need(world_hashes==fit_world_hashes,"candidate current30 self/world identities differ from accepted fit proof")
        log("candidate_world_identity_passed",world_hashes=world_hashes,fit_world_hashes_match=True,source_world_bit_replay=True)

        # Build current-target geometry for the four changed stable muscle rows; hash every row against old packs.
        target_identity_rows=[]; changed_by_pose=[]
        for pose_index,(step,op,npk) in enumerate(zip(STEPS,oldpacks,newpacks)):
            old_xyz,old_surfs,_=ca._pack_surfaces(op,legacy)
            new_xyz,new_surfs,_=ca._pack_surfaces(npk,legacy|{SKIN_KEY})
            need(legacy.issubset(set(old_surfs)) and legacy.issubset(set(new_surfs)) and set(old_surfs)-legacy=={SKIN_KEY} and set(new_surfs)-legacy=={SKIN_KEY},f"target/skin key coverage differs at step {step}")
            changes=[]; signatures={}
            for key in sorted(legacy):
                osig=row_sig(old_xyz,old_surfs[key]);nsig=row_sig(new_xyz,new_surfs[key])
                signatures[f"{key[0]}:{key[1]}"]={"old":osig["sha256"],"current30":nsig["sha256"],"reused_exactly":osig["sha256"]==nsig["sha256"]}
                if osig["sha256"]!=nsig["sha256"]:
                    f0=np.asarray(old_surfs[key]["faces"],dtype=np.int64);f1=np.asarray(new_surfs[key]["faces"],dtype=np.int64)
                    v0=np.unique(f0);v1=np.unique(f1)
                    changes.append({"key":f"{key[0]}:{key[1]}","old":osig,"current30":nsig,
                                    "face_rows_equal":bool(np.array_equal(np.searchsorted(v0,f0),np.searchsorted(v1,f1))) if len(v0)==len(v1) and np.array_equal(v0-v0[0],v1-v1[0]) else False,
                                    "max_abs_position_delta_m":float(np.max(np.abs(np.asarray(old_xyz[v0],dtype=np.float64)-np.asarray(new_xyz[v1],dtype=np.float64)))) if len(v0)==len(v1) else None})
            changed={tuple(map(int,x["key"].split(":"))) for x in changes}
            need(changed==REFRESH_KEYS,f"current30 target identity delta at {step} differs from expected 37/38/39/40/67/68/75/76: {sorted(changed)}")
            target_identity_rows.append(signatures);changed_by_pose.append(changes)
            log("current30_target_row_identity_checked",step=step,reused_native_rows=len(legacy)-len(changed),changed_rows=sorted(f"{s}:{i}" for s,i in changed))
            del old_xyz,old_surfs,new_xyz,new_surfs

        # Recreate the two previously offline-forwarded SCM target rows from identical NHTISS and accepted poses.
        tissue=load_tissue(TISS_SOURCE);tissue_manifest=json.loads(TISS_SOURCE_MAN.read_text())
        need(sha(TISS_SOURCE)=="08c534229c322391ce5c7fb972810ace13bc07cf9d95ff18825561ac2f792ada","offline SCM source payload changed")
        rows={sid:tissue_row(tissue,sid) for sid in (151,152)}
        offline_identity=[]
        for i,(step,op,npk) in enumerate(zip(STEPS,oldpacks,newpacks)):
            pose_hashes={}
            for name,rec,poses in [("old22",old_receipts[i],old_poses[i]),("current30",new_receipts[i],new_poses[i])]:
                pair=[]
                for sid in (151,152):
                    row=rows[sid];p=tissue_forward(row,tissue,poses)
                    sig=row_sig(p,{"faces":row["faces"],"body":0})
                    pair.append({"key":f"51005:{sid}","geometry_sha256":sig["sha256"],"faces":len(row["faces"]),"vertices":len(p)})
                pose_hashes[name]=pair
            need(pose_hashes["old22"]==pose_hashes["current30"],f"offline 151/152 projected target geometry differs at step {step}")
            offline_identity.append({"step":step,**pose_hashes,"exact_reuse":True})
        log("offline_151_152_target_identity_passed",pose_count=8,tissue_payload_sha256=sha(TISS_SOURCE))

        # Refresh only changed rows with the exact predicate; merge them into the accepted full 861-row candidate tables.
        candidate_tables=[];pose_results=[];target_validation_cache=set()
        for i,(step,npk) in enumerate(zip(STEPS,newpacks)):
            xyz,surfs,_=ca._pack_surfaces(npk,legacy|{SKIN_KEY})
            changed_faces={key:np.asarray(surfs[key]["faces"],dtype=np.int64) for key in REFRESH_KEYS}
            skin_records=ca._exact_surface_records(candidate_world[i],compact)
            refreshed=ca._target_intersection_audit(skin_records,changed_faces,np.asarray(xyz,dtype="<f4"),_validated_target_geometry_cache=target_validation_cache)
            need(set(refreshed)==refresh_text,f"changed-target exact audit key coverage differs at {step}")
            table=copy.deepcopy(old_tables[i])
            for key,row in refreshed.items():
                # The exact owner completed conversion/degeneracy validation for every target face.
                row=dict(row);row["degenerate_face_rows"]=[]
                table[key]=row
            need(set(table)==expected_text and len(table)==861,f"merged table coverage differs at {step}")
            need(all(type(r.get("count")) is int and r["count"]==len(r.get("triangle_pairs",[])) and not r.get("degenerate_face_rows") for r in table.values()),f"merged table malformed at {step}")
            digest=ca._baseline_target_pair_table_sha256(table);total=sum(int(x["count"]) for x in table.values())
            refreshed_summary={k:{"count":int(refreshed[k]["count"]),"aabb_candidate_pairs":int(refreshed[k]["aabb_candidate_pairs"]),"triangle_pairs":refreshed[k]["triangle_pairs"]} for k in sorted(refreshed)}
            offline_positions=[tissue_forward(rows[sid],tissue,new_poses[i]) for sid in (151,152)]
            target_positions=np.concatenate([np.asarray(xyz,dtype="<f4"),*offline_positions]).astype("<f4",copy=False)
            full_target_faces={k:np.asarray(surfs[k]["faces"],dtype=np.int64) for k in legacy}
            offbase=len(xyz)
            for sid in (151,152):
                full_target_faces[(51005,sid)]=rows[sid]["faces"]+offbase
                offbase+=len(rows[sid]["positions"])
            pose_results.append({"pose_index":i,"step":step,"target_count":861,"identity_reused_count":853,"refreshed_count":8,
                 "refreshed_rows":refreshed_summary,"total_exact_pair_count":total,"candidate_target_pair_table_sha256":digest,
                 "candidate_world_f32_sha256":world_hashes[i],"target_geometry_f32_sha256":ca._target_geometry_f32_sha256(target_positions,full_target_faces)})
            candidate_tables.append(table)
            log("current30_candidate_target_refresh_done",step=step,total_pairs=total,refreshed_counts={k:int(v["count"]) for k,v in refreshed.items()},table_sha256=digest)
            del xyz,surfs,skin_records
        table_path=OUT/"candidate-target-pair-tables.json.gz"
        with gzip.open(table_path,"wt",encoding="utf-8",compresslevel=6) as f:json.dump(candidate_tables,f,sort_keys=True,separators=(",",":"),allow_nan=False)
        all_target_clear=all(p["total_exact_pair_count"]==0 for p in pose_results)

        # Invariant checks for bed support, ocular opening vertices, skin ABI bindings, and respiration/functional state.
        fixed_bed={int(x["vertex_index"]) for x in json.loads(BASE_SCENE.read_text())["bed"]["support_witnesses"]}
        need(len(fixed_bed)==32,"bed witness count changed")
        need(np.array_equal(fit[list(fixed_bed)].view("<u4"),skin["pos"][list(fixed_bed)].view("<u4")),"candidate moved one or more of 32 bed witnesses")
        current_receipt=json.loads(CURRENT_RECEIPT.read_text())
        ocular_interfaces=current_receipt["mass_geometry_accounting"]["skin_boundary_repair"]["retained_anatomical_interfaces"]
        ocular_ids=sorted({int(v) for item in ocular_interfaces for v in item.get("source_vertex_ids",[])})
        need(ocular_ids and np.array_equal(fit[ocular_ids].view("<u4"),skin["pos"][ocular_ids].view("<u4")),"candidate moved ocular boundary source positions")
        need(np.array_equal(fit[list(range(skin["nv"]))][~np.isin(np.arange(skin["nv"]),ref)].view("<u4"),skin["pos"][~np.isin(np.arange(skin["nv"]),ref)].view("<u4")),"candidate changed unreferenced source positions")
        need(fit_event["self_audits_sha256"] and len(fit_event["self_audits_by_pose"])==8,"candidate self proof absent")
        
        # Persist the refreshed exact evidence before export, whether or not a changed row blocks composition.
        input_paths=[SCRIPT,FIT_REPORT,FIT_NPY,FIT_EVENTS,FIT_TARGETS,FIT_PREFLIGHT,BASE_SKIN,BASE_SKIN_MAN,BASE_SCENE,CURRENT_RECEIPT,CURRENT_NHTISS,CURRENT_NHTISS_MAN,CURRENT_COMPOSITION,CURRENT_NATIVE_REPORT,CURRENT_RUN_DECL,INV,FORWARD,TISS_FORWARD,TISS_SOURCE,TISS_SOURCE_MAN,RESP/"src/numilab_human/resting_respiratory_conforming_field.py",GEOM,PREFLIGHT,ANATOMY,REGISTRATION,CA,CI]
        for step in STEPS:
            input_paths.extend([OLD_CAPTURE/f"step-{step}.mrvpack",OLD_CAPTURE/f"step-{step}.receipt.json",NEW_CAPTURE/f"step-{step}.mrvpack",NEW_CAPTURE/f"step-{step}.receipt.json"])
        input_pins=[pin(p) for p in input_paths]
        target_path_pin=pin(table_path)
        fit_manifest=json.loads(BASE_SKIN_MAN.read_text())
        base_receipt_hash=sha(CURRENT_RECEIPT)
        body_pose_state_sha=sha_bytes(json.dumps([{"step":s,"body_poses":r["accepted_registered_body_poses"],"resp":r["accepted_respiratory_motion"]} for s,r in zip(STEPS,new_receipts)],sort_keys=True,separators=(",",":")).encode())
        report={"schema":"numi.human.skin-candidate-current30-pose-target-refresh.v1",
          "status":"candidate_target_refresh_complete_pending_composition" if all_target_clear else "candidate_target_refresh_failed_nonzero_pairs",
          "qualification_boundary":"Offline accepted 30-surface capture-pose evidence only; no native run launched. This is not a whole-cycle or completed anatomy/interface qualification.",
          "native_run_performed":False,"gpu_run_performed":False,"candidate_admitted":False,
          "candidate_skin_fit":{"report":pin(FIT_REPORT),"full_source_npy":pin(FIT_NPY),"full_source_f32_sha256":f32sha(fit),"accepted_event_line_sha256":sha_bytes((json.dumps(fit_event,sort_keys=True,separators=(",",":"))+"\n").encode()),
             "accepted_event_target_audits":pin(FIT_TARGETS),"source_displacement_max_mm":float(fit_report["maximum_source_displacement_m"]*1000),"all8_self_pair_counts_zero":True},
          "captures":{"steps":STEPS,"current30_composition":pin(CURRENT_COMPOSITION),"native30_verification":pin(CURRENT_NATIVE_REPORT),"old22_and_current30_receipts_match_registered_pose_and_respiration":True,"body_pose_count":157,"old_new_skin_capture_world_f32_equal":True,"baseline_915_source_forward_bit_replay":True,"candidate_world_hashes_match_fit_self_proof":True,"candidate_world_f32_sha256_by_pose":world_hashes,"body_pose_respiration_state_sha256":body_pose_state_sha},
          "target_refresh":{"legacy_target_count":859,"offline_target_count":2,"full_target_count":861,"refreshed_rows":[f"{s}:{i}" for s,i in sorted(REFRESH_KEYS)],"refreshed_row_count":8,"reused_exact_identity_row_count":853,
             "reused_rows_basis":"For each of eight steps, all other 851 packed target rows have exact canonical local-face/F32-position identity between old22 and current30 packs; offline 151/152 target geometry was recomputed from the pinned NHTISS4 source and matching accepted body poses and exactly matched old22/current30. Candidate skin-world F32 hashes match the accepted fit proof.",
             "per_pose_target_identities":target_identity_rows,"per_pose_changed_rows":changed_by_pose,"offline_151_152_pose_identity":offline_identity,
             "candidate_tables_path":str(table_path),"candidate_tables":target_path_pin,"poses":pose_results,"all861_target_pairs_zero_all8":all_target_clear},
          "preservation":{"bed_support_witness_count":32,"bed_witness_positions_bit_exact":True,"ocular_boundary_source_vertex_count":len(ocular_ids),"ocular_boundary_positions_bit_exact":True,
              "unreferenced_source_vertices_bit_exact":True,"skin_topology_face_indices_preserved":True,"skin_binding_records_and_weights_preserved_by_composer":True,
              "current30_NHTISS_and_resting_anatomy_functional_bindings_preserved_by_composer":True,"respiratory_state_capture_unchanged":True},
          "inputs":input_pins}
        write_json(OUT/"target-refresh-report.json",report)
        need(all_target_clear,"current30 refreshed candidate target audit has nonzero exact pairs; refusing NHSKIN/receipt composition")

        # Export candidate with the existing NHSKIN position-composition owner.
        vertex_offset=skin["vertex_offset"];edited=bytearray(skin_raw)
        for v in np.flatnonzero(np.any(fit.view("<u4")!=skin["pos"].view("<u4"),axis=1)):
            struct.pack_into("<3f",edited,vertex_offset+int(v)*56,*map(float,fit[v]))
        new_payload,owner=geom.compose_disjoint_skin_position_corrections(skin_raw,[bytes(edited)],global_source_matrix=json.loads(REGISTRATION.read_text())["coordinate_system"]["global_source_mm_to_myosim_world_m"])
        decoded=pre.decode_payload(new_payload)
        need(np.array_equal(decoded["vertices_u"][:,:3].view("<f4"),fit.view("<f4")),"exported NHSKIN positions differ from accepted fit NPY")
        need(new_payload[:vertex_offset]==skin_raw[:vertex_offset] and new_payload[skin["vertex_end"]:]==skin_raw[skin["vertex_end"]:],"NHSKIN export changed header/bindings/weights/topology")
        asset=OUT/"bodyparts3d-myosim-skinned-shell.nhskin";asset.write_bytes(new_payload)
        manifest=copy.deepcopy(fit_manifest)
        manifest["inputs"]["source_payload"]={**pin(BASE_SKIN),"route":"accepted result-014 source-position candidate; exact target rows refreshed against current30 captures"}
        manifest["inputs"]["upstream_skin_provenance"]={"immediate_source_path":str(BASE_SKIN_MAN),"immediate_source_sha256":sha(BASE_SKIN_MAN),"immediate_source_payload":pin(BASE_SKIN)}
        manifest["output_payload"]=pin(asset);manifest["output_manifest"]=str((OUT/"common-atlas-skin-geometry-registration.manifest.json").resolve())
        manifest["geometry_registration"]={**manifest.get("geometry_registration",{}),"method":"Existing NHSKIN source-position composition owner after current30 exact target refresh",
             "candidate_source_positions":pin(FIT_NPY),"fit_report":pin(FIT_REPORT),"current30_target_refresh_report":pin(OUT/"target-refresh-report.json"),
             "current30_candidate_target_pair_tables":pin(table_path),"owner_composition":owner,
             "source_revisions":{"clearance_owner_sha256":sha(CA),"exact_predicate_sha256":sha(CI),"skin_composition_owner_sha256":sha(GEOM),"receipt_composition_owner_sha256":sha(ANATOMY)},
             "interpretation":"Inferred engineering reference geometry; not measured participant tissue thickness."}
        manifest["method"]="Existing NHSKIN source-position composition owner after the accepted 8-pose fit and exact current30 target-row refresh."
        manifest["evidence_boundary"]="Offline reconstruction at eight retained current30 poses. No native replay or fresh whole-cycle validation. SCM vessel-interface correction is still open."
        manifest["qualification"]={**manifest.get("qualification",{}),"current30_candidate_target_refresh":"all 861 exact target tables at eight captured poses; eight changed target rows freshly audited","native_skin_candidate":"pending native replay and combined SCM-vessel checks","whole_body_anatomy":"not qualified"}
        manifest["candidate_status"]="pending_native_replay_and_combined_SCM_vessel_clearance"
        # Bind this candidate manifest to the current30 NHTISS4 owner identity before receipt composition.
        reg_doc=json.loads(REGISTRATION.read_text())
        canonical=manifest["inputs"]["canonical_binding_reference"]
        nhtiss_identity=anatomy._verify_nhtiss_common_atlas_source_identity(
            tissue_payload_path=CURRENT_NHTISS,tissue_manifest_path=CURRENT_NHTISS_MAN,
            registration_sha256=manifest["inputs"]["registration_sha256"],
            source_archive_sha256=manifest["payload_identity"]["source_archive_sha256"],
            registration_fingerprint32=int(manifest["payload_identity"]["registration_fingerprint32"],16),
            rigid_payload_sha256=current_receipt["provenance"]["rigid_payload_sha256"],candidate_skin=new_payload,
            canonical_reference_path=Path(canonical["path"]),canonical_reference_sha256=canonical["sha256"],registration=reg_doc)
        manifest["nhtiss4_common_atlas_source_identity"]=nhtiss_identity
        manifest["mass_geometry_accounting_nhtiss4_source_identity_sha256"]=sha_bytes(json.dumps(nhtiss_identity,sort_keys=True,separators=(",",":")).encode())
        manifest_path=OUT/"common-atlas-skin-geometry-registration.manifest.json"
        write_json(manifest_path,manifest)
        # Stage the unchanged current30 receipt and its standard companion manifest for the existing owner.
        staged_base=OUT/"base-receipt-input";staged_base.mkdir()
        staged_receipt=staged_base/"resting-anatomy-receipt.json";staged_receipt.write_bytes(CURRENT_RECEIPT.read_bytes())
        base_manifest={"schema":"numi.human.resting-anatomy-manifest.v1","payload":copy.deepcopy(current_receipt["payload"]),
            "receipt":{"path":str(staged_receipt.resolve()),"sha256":sha(staged_receipt)},
            "functional_bindings":copy.deepcopy(current_receipt.get("functional_bindings")),
            "qualification":copy.deepcopy(current_receipt.get("qualification")),
            "native_muscle_surfaces":copy.deepcopy(current_receipt.get("provenance",{}).get("native_muscle_surfaces")),
            "thorax_source_volume_m3":current_receipt.get("thorax_source_volume_m3"),
            "source_surfaces":copy.deepcopy(current_receipt.get("provenance",{}).get("source_id_map")),
            "mass_geometry_accounting":copy.deepcopy(current_receipt.get("mass_geometry_accounting"))}
        write_json(staged_base/"resting-anatomy-manifest.json",base_manifest)
        common=current_receipt.get("provenance",{}).get("cardiac_geometry_binding",{}).get("common_field",{})
        for field in ("map","polynomials","domain_boxes"):
            rec=common.get(field)
            if isinstance(rec,dict):
                rel=Path(rec["path"]);src=rel if rel.is_absolute() else CURRENT/rel
                need(src.is_file() and sha(src)==rec["sha256"],f"current30 cardiac sidecar identity mismatch: {field}")
                anatomy._copy_checked_sidecar(src,staged_base/src.name,rec["sha256"])
        # Bind the exported skin into the actual current30 anatomy receipt through the existing receipt owner.
        composed_dir=OUT/"composed-anatomy"
        composed=anatomy.compose_skin_binding_candidate(staged_receipt,asset,manifest_path,composed_dir)
        final_receipt=Path(composed["receipt_path"]);final_manifest=final_receipt.with_name("resting-anatomy-manifest.json")
        final_doc=json.loads(final_receipt.read_text());base_doc=json.loads(CURRENT_RECEIPT.read_text())
        need(final_doc["payload"]==base_doc["payload"],"anatomy payload changed during skin composition")
        need(final_doc["functional_bindings"]==base_doc["functional_bindings"],"functional binding state changed during skin composition")
        need(final_doc["provenance"]["respiratory_configuration"]==base_doc["provenance"]["respiratory_configuration"],"respiratory configuration changed during skin composition")
        need(final_doc["provenance"]["native_muscle_surfaces"]==base_doc["provenance"]["native_muscle_surfaces"],"current30 NHTISS provenance changed during skin composition")
        need(final_doc["mass_geometry_accounting"]["skin_payload_sha256"]==sha(asset),"receipt does not bind exported skin")
        need(final_doc["provenance"]["skin_visual_binding_candidate"]["preservation"]["all_86_canonical_binding_records_byte_identical"] is True,"receipt composer did not preserve all skin binding records")
        scene=json.loads(BASE_SCENE.read_text());scene["source"]["skin"].update(path=str(asset),sha256=sha(asset))
        scene_path=OUT/"resting-supine-scene.manifest.json";write_json(scene_path,scene)
        # Prepare a non-executable declaration preview based on the exact current30 declaration.
        decl=json.loads(CURRENT_RUN_DECL.read_text());argv=list(decl["argv"])
        next_output=OUT/"native-run-not-launched"
        def setarg(flag,val):
            idx=argv.index(flag);argv[idx+1]=str(val)
        setarg("--body-scene",scene_path);setarg("--anatomy-receipt",final_receipt);setarg("--output",next_output)
        for i,v in enumerate(argv):
            if isinstance(v,str) and v.startswith("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT="):
                argv[i]="NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT="+str(next_output/"common-field-failure.json")
        decl["argv"]=argv;decl["immutable_assets"].update({str(asset):sha(asset),str(manifest_path):sha(manifest_path),str(final_receipt):sha(final_receipt),str(final_manifest):sha(final_manifest),str(scene_path):sha(scene_path),str(table_path):sha(table_path),str(OUT/"target-refresh-report.json"):sha(OUT/"target-refresh-report.json")});decl["immutable_assets"].pop(str(CURRENT_RECEIPT),None)
        decl["qualification"]="Preview only. Candidate skin was refreshed offline against eight changed target rows across eight current30 captured poses; native replay and the combined SCM-vessel correction remain pending. Do not launch from this preview."
        decl["candidate_preview"]={"status":"prepared_not_launched","native_run_performed":False,"gpu_run_performed":False,"output_path_reserved_not_created":str(next_output),"blocking_gates":["combined SCM-vessel clearance and attachment review","fresh native replay of candidate skin and current30 muscle rows","whole-cycle skin/self/target validation"]}
        decl_path=OUT/"run-declaration-preview.json";write_json(decl_path,decl)

        # Rehash every declared input after the work.
        post=[]
        for item in input_pins:
            p=Path(item["path"]);post.append({**item,"post_sha256":sha(p),"unchanged":sha(p)==item["sha256"]})
        need(all(x["unchanged"] for x in post),"one or more source inputs changed during preparation")
        final_report={**report,"status":"skin_candidate_composed_pending_native_and_SCM_vessel_checks","candidate_skin":pin(asset),"candidate_skin_manifest":pin(manifest_path),
          "final_resting_anatomy_receipt":pin(final_receipt),"final_resting_anatomy_manifest":pin(final_manifest),"candidate_scene_manifest":pin(scene_path),
          "run_declaration_preview":pin(decl_path),"receipt_composition":composed,"skin_owner_composition":owner,
          "all_declared_inputs_unchanged":True,"input_postcheck":post,"native_run_performed":False,"gpu_run_performed":False,"candidate_admitted":False}
        write_json(OUT/"preparation-report.json",final_report)
        log("composition_prepared_no_native_run",skin_sha256=sha(asset),receipt_sha256=sha(final_receipt),declaration_sha256=sha(decl_path))
        return 0
    except Exception as e:
        failure={"status":"failed_closed","error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),"native_run_performed":False,"gpu_run_performed":False}
        try:write_json(OUT/"failure.json",failure)
        except Exception:pass
        raise
if __name__=="__main__":raise SystemExit(main())
