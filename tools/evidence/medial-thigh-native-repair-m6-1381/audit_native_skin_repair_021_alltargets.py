#!/usr/bin/env python3
"""Exact CPU changed-star skin clearance audit against the complete registered target inventory."""
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np

A=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276")
NBASE=A/"back-forearm-native-014-018-back412-recovery-16s/baseline/native-run"
NCAND=A/"back412-skin-reference-native-021-003/baseline/native-run"
PUB=Path("/Users/numi/numi-human-left-pq-publication-1344/src")
INV=Path("/Users/numi/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
OUT=A/"full-anatomy-delivery-audit-1378"
STEPS=[0,3807,4767,5023,5599,6207,7423,8000]
SKIN_KEY=(51007,1)
N18_SKIN_SHA="24f0d0bf288bb501ee7f0464f53e089126c4d4850ad638044b0015a286497d6a"
N21_SKIN_SHA="b01028f163bde0d46c9605d730f2e104713a1c490be3c718e77ff9cebb865710"
NHA_SHA="1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
NHTISS_SHA="3aa1dcf1458f6c7cda617a63ae4b4ed37a3280128da0b3dd3f1af571a0ce56d4"
CA_SHA="fccc7f29d421c9ad8e7da40d61d16cc8633244a16829dc7a7cd9d83ff109cfae"
CI_SHA="05ba7856fc373f55a1270eb5f3ef887a4eac493ad9fb4590a618caee952d2a32"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def load(p): return json.loads(Path(p).read_text())

def pose_map(receipt):
    return {int(x["body_index"]):(np.asarray(x["position_m"],dtype="<f4"),np.asarray(x["quaternion_xyzw"],dtype="<f4"))
            for x in receipt["accepted_body_poses"]}

def audit_rows(ca,ci,skin_records,target_faces,positions):
    exact_index=ca._prepare_surface_aabb(skin_records)
    float_index=ca._float32_skin_aabb_index(skin_records)
    counts={}; broad=0; tested=0
    for key in sorted(target_faces):
        faces=np.asarray(target_faces[key],dtype=np.int64)
        rows,aabb_count=ca._candidate_target_rows_for_skin_aabb(faces,positions,float_index)
        broad+=int(aabb_count)
        if not rows: continue
        chosen=faces[np.asarray(rows,dtype=np.int64)]
        unique=np.unique(chosen)
        compact=np.searchsorted(unique,chosen)
        target_records=ca._exact_surface_records(positions[unique],compact)
        target_records=[(r[0],r[1],r[2],int(rows[i]),r[4]) for i,r in enumerate(target_records)]
        audit=ci._audit_pair_prepared_first(exact_index,target_records)
        if int(audit["aabb_candidate_pairs"])!=int(aabb_count):
            raise AssertionError(f"target broadphase mismatch {key}: {audit['aabb_candidate_pairs']} != {aabb_count}")
        n=int(audit["count"]); tested+=len(rows)
        if n: counts[f"{key[0]}:{key[1]}"]={"count":n,"target_triangle_rows":len(rows),"aabb_pairs":int(aabb_count)}
    return counts,{"aabb_candidate_pairs":broad,"target_triangle_rows_exactly_tested":tested}

def main():
    sys.path.insert(0,str(PUB))
    from numilab_human import common_atlas_skin_clearance as ca
    from numilab_human import cardiac_cavity_intersections as ci
    if sha(Path(ca.__file__))!=CA_SHA or sha(Path(ci.__file__))!=CI_SHA: raise AssertionError("pinned predicate source changed")
    inventory,_,expected_counts,populations=ca._load_target_inventory(INV)
    all_keys=set(inventory)|{(51005,sid) for sid in range(151,165)}
    assert len(inventory)==859 and len(all_keys)==873
    n18=load(NBASE/"run-metadata.json"); n21=load(NCAND/"run-metadata.json")
    def check_run(run,skinsha):
        names={Path(p).name:d for p,d in run["asset_sha256"].items()}
        assert names["bodyparts3d-myosim-skinned-shell.nhskin"]==skinsha
        assert names["resting-thorax.nhanatomy"]==NHA_SHA
        assert names["bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"]==NHTISS_SHA
        assert run["host"]=="numi.local" and run["exit_code"]==0
        assert all(Path(p).is_file() and sha(p)==d for p,d in run["asset_sha256"].items())
    check_run(n18,N18_SKIN_SHA); check_run(n21,N21_SKIN_SHA)
    captures=[]; start=time.monotonic()
    for step in STEPS:
        bpack=NBASE/"accepted-geometry"/f"step-{step}.mrvpack"
        cpack=NCAND/"accepted-geometry"/f"step-{step}.mrvpack"
        br=load(NBASE/"accepted-geometry"/f"step-{step}.receipt.json")
        cr=load(NCAND/"accepted-geometry"/f"step-{step}.receipt.json")
        assert br["accepted_step"]==cr["accepted_step"]==step
        assert br["physical_endpoint"]==cr["physical_endpoint"]=="accepted"
        assert br["surface_audit_endpoint"]==cr["surface_audit_endpoint"]=="passed"
        assert sha(bpack)==br["pack_file_sha256"] and sha(cpack)==cr["pack_file_sha256"]
        bp,bs,_=ca._pack_surfaces(bpack,all_keys)
        cp,cs,_=ca._pack_surfaces(cpack,all_keys)
        assert set(all_keys).issubset(bs) and set(all_keys).issubset(cs)
        bskin=bs[SKIN_KEY]; cskin=cs[SKIN_KEY]
        bf=np.asarray(bskin["faces"],dtype=np.int64); cf=np.asarray(cskin["faces"],dtype=np.int64)
        assert bf.shape==cf.shape and np.array_equal(bf,cf)
        bbase=int(br["skin_source_mapping"]["pack_first_vertex"]); cbase=int(cr["skin_source_mapping"]["pack_first_vertex"])
        assert int(br["skin_source_mapping"]["source_vertex_count"])==int(cr["skin_source_mapping"]["source_vertex_count"])
        bn=int(br["skin_source_mapping"]["source_vertex_count"]); cn=int(cr["skin_source_mapping"]["source_vertex_count"])
        assert np.array_equal(bf-bbase,cf-cbase)
        bx=np.asarray(bp[bbase:bbase+bn],dtype="<f4")
        cx=np.asarray(cp[cbase:cbase+cn],dtype="<f4")
        assert bx.shape==cx.shape and np.isfinite(bx).all() and np.isfinite(cx).all()
        changed=np.flatnonzero(np.any(bx.view("<u4")!=cx.view("<u4"),axis=1))
        # Candidate source repair edits 52 source vertices; unchanged body poses must localize actual capture differences.
        bposes=pose_map(br); cposes=pose_map(cr)
        assert set(bposes)==set(cposes) and len(bposes)==len(cposes)==157
        pose_differences=[]
        for body in sorted(bposes):
            if not (np.array_equal(bposes[body][0].view("<u4"),cposes[body][0].view("<u4"))
                    and np.array_equal(bposes[body][1].view("<u4"),cposes[body][1].view("<u4"))):
                pose_differences.append(body)
        if pose_differences: raise AssertionError(f"native baseline and repair poses differ at step {step}: {pose_differences[:12]}")
        if len(changed)>128: raise AssertionError(f"skin difference escaped bounded repair support at step {step}: {len(changed)} vertices")
        star_rows=np.flatnonzero(np.any(np.isin(cf-cbase,changed),axis=1))
        star_faces=(cf[star_rows]-cbase).astype(np.int64)
        star_vertices=np.unique(star_faces)
        star_local=np.searchsorted(star_vertices,star_faces)
        records=ca._exact_surface_records(cx[star_vertices],star_local)
        target_faces={k:np.asarray(cs[k]["faces"],dtype=np.int64) for k in sorted(all_keys)}
        nonzero,broad=audit_rows(ca,ci,records,target_faces,cp)
        result={"step":step,"time_s":cr["accepted_time_s"],"native_pack_sha256":cr["pack_file_sha256"],
            "target_inventory_count":len(all_keys),"source_inventory_count":len(inventory),
            "added_muscle_target_count":14,"native_skin_changed_vertices_vs_N18":len(changed),
            "changed_vertex_ids":changed.astype(int).tolist(),"affected_skin_face_count":len(star_rows),
            "affected_skin_unique_vertices":len(star_vertices),"candidate_changed_star_exact_target_pairs":sum(x["count"] for x in nonzero.values()),
            "nonzero_target_rows":nonzero,"broadphase":broad}
        captures.append(result)
        progress={"schema":"numi.human.native-021-skin-changed-star-all-targets.v1",
            "status":"running","scope":"Exact Float32 CPU triangle intersections for the captured 021 skin faces touching vertices changed relative to N18, against all 859 complete inventory surfaces plus 14 appended muscles (873 total). Full skin self and 92 selected pairs are separately covered by native-skin-repair-021-audit.json.",
            "inputs":{"N18_skin_sha256":N18_SKIN_SHA,"native021_skin_sha256":N21_SKIN_SHA,"NHA_sha256":NHA_SHA,"NHTISS_sha256":NHTISS_SHA,
                "N18_binary_sha256":"147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174","native021_binary_sha256":"147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174",
                "clearance_module_sha256":CA_SHA,"predicate_module_sha256":CI_SHA},
            "inventory":{"859_complete_surface_rows":len(inventory),"14_appended_muscle_rows":14,"total_target_surfaces":len(all_keys),"category_counts":populations,
                "legacy_exact_overlap_witness_count":sum(expected_counts.values())},
            "capture_count":len(captures),"captures":captures,"elapsed_s":time.monotonic()-start}
        (OUT/"native-skin-repair-021-alltarget-progress.json").write_text(json.dumps(progress,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"step":step,"changed_vertices":len(changed),"changed_star_faces":len(star_rows),"targets":len(all_keys),
            "pairs":result["candidate_changed_star_exact_target_pairs"],"nonzero_targets":len(nonzero),"elapsed_s":time.monotonic()-start},sort_keys=True),flush=True)
        del bp,bs,cp,cs,bx,cx,records,target_faces,nonzero
        gc.collect()
    final=load(OUT/"native-skin-repair-021-alltarget-progress.json")
    final["status"]="complete"
    final["complete"]=True
    final["all8_changed_star_target_pairs_zero"]=all(not r["nonzero_target_rows"] for r in captures)
    final["elapsed_s_total"]=time.monotonic()-start
    for path,digest in {**n18["asset_sha256"],**n21["asset_sha256"]}.items():
        if not Path(path).is_file() or sha(path)!=digest: raise AssertionError(f"input changed after audit: {path}")
    final["input_pins_revalidated"]=True
    (OUT/"native-skin-repair-021-alltarget-audit.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"complete":True,"all8_zero":final["all8_changed_star_target_pairs_zero"],"inventory":len(all_keys),"elapsed_s_total":final["elapsed_s_total"]},sort_keys=True),flush=True)

if __name__=="__main__": main()
