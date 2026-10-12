#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, shutil, time
from pathlib import Path

PUB=Path("/Users/n/numi-human-left-pq-publication-1344/tools/evidence/bilateral-forearm-preparation-m6-1360")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
F=Path("/Users/n/numi-human-fds132-m6-1350")
P=Path("/Users/n/numi-human-left-pq-publication-1344/tools/evidence/native-anatomy-left-forearm-diagonal-m6-1358")
C=A/"bilateral-forearm-composition-preflight-013-003"
C2=A/"bilateral-forearm-composition-preflight-013-002"
N=A/"bilateral-forearm-native-013-canonical150"
N012=A/"forearm-shared-field-native-012-diagonal097"
B=F/"attempt-156-right-candidate150-carrier-assets-001"
S=F/"attempt-155-right-candidate150-carrier-assets-001"
C150=F/"attempt-150-right-canonical-midpoint-field-correction-001/result-001"
R153=F/"attempt-153-candidate122-exact-route-field-replay-001"
R154=F/"attempt-154-exact-source-route-field-replay-001"
L157=F/"attempt-157-right-candidate150-late151999-f64-critical-001"
R118=F/"attempt-118-right-paired-cap-short-edge-contractions-001/result-001"
R121=F/"attempt-121-right-witness-band-guarded-contractions-001/result-001"
R122=F/"attempt-122-right-cap-boundary-directed-contractions-001/result-001"
BASELINE_LATE=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011/baseline/native-run/accepted-geometry")

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def jread(p): return json.loads(Path(p).read_text())
def jwrite(p,x):
    p=PUB/p if not Path(p).is_absolute() else Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")

def files():
    return {
      "proofs/candidate150/construction-checkpoint.json":C150/"construction-checkpoint.json",
      "proofs/candidate150/summary.json":C150/"summary.json",
      "proofs/candidate150/source-critical.json":C150/"source-critical.json",
      "proofs/candidate150/current8000-f64-critical.json":C150/"current8000-f64-critical.json",
      "proofs/candidate150/current8000-f32-separate-critical.json":C150/"current8000-f32-separate-critical.json",
      "proofs/candidate150/current8000-f32-fma-critical.json":C150/"current8000-f32-fma-critical.json",
      "proofs/candidate150/new-midpoint-rest-check.json":C150/"new-midpoint-rest-check.json",
      "proofs/replay/attempt153-quantized-coordinate-diagnostic.json":R153/"report.json",
      "proofs/replay/attempt154-exact-rational-coordinate-replay.json":R154/"report.json",
      "proofs/late151999/attempt157-f64-critical.json":L157/"report.json",
      "proofs/bundle156/preparation.json":B/"preparation.json",
      "proofs/bundle156/construction-delta-104-to-122.json":S/"construction-delta-104-to-122.json",
      "proofs/bundle156/construction-delta-122-to-150.json":S/"construction-delta-122-to-150.json",
      "proofs/bundle156/construction-chain/attempt-118-construction-checkpoint.json":R118/"construction-checkpoint.json",
      "proofs/bundle156/construction-chain/attempt-121-construction-report.json":R121/"construction-report.json",
      "proofs/bundle156/construction-chain/attempt-122-construction-report.json":R122/"construction-report.json",
      "proofs/composition013-003/preparation.json":C/"preparation.json",
      "proofs/composition013-003/composition-report.json":C/"assets/report.json",
      "proofs/composition013-003/rest-registration-preflight.json":C/"rest-registration-preflight.json",
      "proofs/composition013-003/surface-manifest.json":C/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json",
      "proofs/composition013-003/row-131-report.json":C/"patches/row-131.json",
      "proofs/composition013-003/row-133-report.json":C/"patches/row-133.json",
      "proofs/composition013-002/row-131-report.json":C2/"patches/row-131.json",
      "proofs/composition013-002/row-133-report.json":C2/"patches/row-133.json",
      "execution/native013/preparation.json":N/"preparation.json",
      "execution/native013/run-declaration.json":N/"baseline/run-declaration.json",
      "execution/native013/launch-command.txt":N/"launch-command.txt",
      "execution/native013/launch-guard.json":N/"launch-guard.json",
      "execution/native013/launch_arm.py":N/"launch_arm.py",
      "execution/native013/composition-report.json":N/"assets/report.json",
      "execution/prior-native012/summary.json":P/"summary.json",
      "execution/prior-native012/run-declaration.json":P/"execution/run-declaration.json",
      "execution/prior-native012/execution.json":P/"execution/execution.json",
      "provenance/registered-route-transform.json":F/"registration-source-b1b410ad.json",
      "provenance/source-route-controls-061.json":F/"attempt-050/arrangement-001/diagnostic-061/fds-fdp-source-routes-061.json",
      "provenance/anatomy-source-attribution-1358.json":P/"anatomy-source-attribution.json",
      "scripts/attempt-095-right-registered-route-field-001.py":F/"attempt-095-right-registered-route-field-001.py",
      "scripts/attempt-150-right-canonical-midpoint-field-correction-001.py":F/"attempt-150-right-canonical-midpoint-field-correction-001.py",
      "scripts/replay-attempt153.py":R153/"replay_candidate122.py",
      "scripts/replay-attempt154.py":R154/"replay_exact_sourcepoints.py",
      "scripts/audit-attempt157.py":F/"audit_right_candidate150_late151999_f64_157.py",
      "scripts/attempt155-carrier-assets-first-pass.py":F/"attempt-155-right-candidate150-carrier-assets-001.py",
      "scripts/finalize-bundle156.py":F/"finalize_right_candidate150_bundle_156.py",
      "scripts/prepare-composition013-001.py":A/"prepare_bilateral_forearm_composition_013_001.py",
      "scripts/prepare-composition013-002.py":A/"prepare_bilateral_forearm_composition_013_002.py",
      "scripts/prepare-native013.py":A/"prepare_bilateral_native_013.py",
      "scripts/audit-native013.py":A/"audit_bilateral_native_013.py",
      "scripts/audit-external-native013.py":A/"audit_bilateral_external_native_013.py",
      "scripts/trace-source-routes-061.py":F/"attempt-050/arrangement-001/diagnostic-061/trace_fds_fdp_source_routes_061.py",
    }

def put_pin(pins,p,d,who):
    if not isinstance(p,str) or not p.startswith("/") or not isinstance(d,str) or len(d)!=64: return
    rec=pins.setdefault(p,{"expected":[],"declared_by":[]})
    if d not in rec["expected"]: rec["expected"].append(d)
    rec["declared_by"].append(who)

def scan_pins(x,pins,who):
    if isinstance(x,dict):
        for k in ("input_sha256","inputs_sha256","input_hashes","immutable_assets"):
            if isinstance(x.get(k),dict):
                for p,d in x[k].items(): put_pin(pins,p,d,who+":"+k)
        paths={k[:-5]:v for k,v in x.items() if k.endswith("_path") and isinstance(v,str)}
        for k,v in x.items():
            p=paths.get(k[:-7]) if k.endswith("_sha256") else None
            if isinstance(v,str) and p and not Path(p).is_dir():
                put_pin(pins,p,v,who+":"+k[:-7])
        if isinstance(x.get("path"),str) and isinstance(x.get("sha256"),str):
            put_pin(pins,x["path"],x["sha256"],who+":path")
        for v in x.values(): scan_pins(v,pins,who)
    elif isinstance(x,list):
        for v in x: scan_pins(v,pins,who)

def main():
    if not PUB.is_dir(): raise RuntimeError("target publication directory missing")
    t0=time.monotonic()
    srcfiles=files()
    copied=[]
    for rel,src in srcfiles.items():
        if not src.is_file(): raise FileNotFoundError(src)
        dst=PUB/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
        sh,dh=sha(src),sha(dst)
        copied.append({"archived_path":rel,"archived_sha256":dh,"source_path":str(src),
                       "source_sha256":sh,"gzip":False,"size_bytes":dst.stat().st_size,
                       "copy_hash_match":sh==dh})
    docs=[C150/"summary.json",C150/"construction-checkpoint.json",R153/"report.json",R154/"report.json",
          R118/"construction-checkpoint.json",R121/"construction-report.json",R122/"construction-report.json",
          L157/"report.json",B/"preparation.json",C/"preparation.json",C/"assets/report.json",
          N/"preparation.json",N/"baseline/run-declaration.json",P/"preparation.json",
          P/"execution/run-declaration.json",P/"execution/execution.json"]
    pins={}
    for d in docs:
        if d.is_file(): scan_pins(jread(d),pins,str(d))
    bundle=jread(B/"preparation.json"); late=jread(L157/"report.json")
    current=jread(C150/"current8000-f64-critical.json")
    comp=jread(C/"preparation.json"); native=jread(N/"preparation.json")
    manual={
      F/"attempt-104-right-witness-band-refinement-001/result-001/candidate.pkl":("28ab429feb10615a5cfbd2345148d6b0619a7e75b41301b3f86f0a244202ad6d","Candidate104 lineage parent"),
      F/"attempt-118-right-paired-cap-short-edge-contractions-001/result-001/candidate.pkl":("cfd81debd3317b076bddc82a6f58c5490748a02a100370d4958697c08775a2a2","Candidate118 after 16 construction contractions"),
      F/"attempt-121-right-witness-band-guarded-contractions-001/result-001/candidate.pkl":("6f60f73521017f3e0f3e5b35e89dd78d29c88d29303a97f5f71950eb7e4548ce","Candidate121 after 2 construction contractions"),
      F/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/candidate.pkl":("aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94","Candidate122 structural parent"),
      F/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/candidate.pkl":(bundle["candidate"]["candidate150_sha256"],"Candidate150 unadmitted source candidate"),
      F/"attempt-095-right-registered-route-field-001/result-001/right-route-anchor-reference-field-001.npz":("7f9c61b6ad804ab859f29fa44fc7d2ea2bc884f7b8468b441749c02648fc6146","Canonical route field input"),
      BASELINE_LATE/"step-151999.mrvpack":(late["pose"]["pack_sha256"],"Late151999 accepted-geometry pack"),
      BASELINE_LATE/"step-151999.receipt.json":(late["pose"]["receipt_sha256"],"Late151999 accepted-geometry receipt"),
      N012/"baseline/native-run/accepted-geometry/step-8000.mrvpack":(current["pose_pack_sha256"],"Current8000 accepted-geometry pack"),
      N012/"baseline/native-run/accepted-geometry/step-8000.receipt.json":(current["pose_receipt_sha256"],"Current8000 accepted-geometry receipt"),
      N/"resting-anatomy-receipt.json":("96b3c4c227c28bcc80f6279bfd89d0cae9fd2f98eb074d6933ff90031d68bb1b","Large native013 anatomy receipt"),
      N/"passive-tissue-parent-bindings.json":(native["extension_sha256"],"Full21 bilateral runtime parent extension"),
      N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue":(native["child_payload_sha256"],"Composed full-body child payload"),
      N/"assets/left-full21-carriers.nhtissue":("ac435503a9b0826f79a4b95a1b0be6d656114b3378239b210b06c4dcf54e54f6","Left full21 load-only carriers"),
      N/"assets/right-full21-carriers.nhtissue":(bundle["carrier_payload"]["sha256"],"Right full21 load-only carriers"),
      N/"common-cardiac-map-f32.bin":("c0b0811fe2e2fb45dc37201205e36af39cf092739cfb52c56fdbad9a67dbb858","Native common cardiac map"),
      C2/"patches/row-131.npz":("d906270471fcd3dae479152a4d876381a2e7cebe917241710fe725986a09b6ad","Interrupted composition013-002 patch"),
      C2/"patches/row-133.npz":("6d5973cc2e0acdb6fc513314723958683ccc4dd54b9efde3a23cb09f477d2f5c","Interrupted composition013-002 patch"),
    }
    for sid in ("131","133"):
        r=bundle["geometry_rows"][sid]
        manual[Path(r["path"])]=(r["sha256"],f"Candidate150 stable{sid} geometry/field NPZ")
    manual[Path(bundle["carrier_payload"]["path"])]=(bundle["carrier_payload"]["sha256"],"Candidate150 right carrier payload")
    manual[Path(bundle["parent_terms"]["source_index0_path"])]=(bundle["parent_terms"]["source_index0_sha256"],"Candidate150 source-index0 parent terms")
    manual[Path(bundle["parent_terms"]["native013_source_index1_path"])]=(bundle["parent_terms"]["native013_source_index1_sha256"],"Candidate150 native013 source-index1 parent terms")
    for r in comp["composition"]["changed_rows"]:
        manual[Path(r["candidate"])]=(r["candidate_sha256"],f"Composition013-003 row{r['stable_id']} patch")
    for p,(d,reason) in manual.items(): put_pin(pins,str(p),d,"explicit:"+reason)
    pin_rows=[]
    copied_sources={str(s) for s in srcfiles.values()}
    for p,r in sorted(pins.items()):
        q=Path(p); exists=q.is_file(); actual=sha(q) if exists else None
        expected=sorted(set(r["expected"]))
        status="verified" if exists and len(expected)==1 and actual==expected[0] else ("missing" if not exists else "mismatch")
        pin_rows.append({"path":p,"expected_sha256":expected,"actual_sha256":actual,
                         "size_bytes":q.stat().st_size if exists else None,"status":status,
                         "declared_by":sorted(set(r["declared_by"]))})
    verified=sum(r["status"]=="verified" for r in pin_rows)
    pin_status="verified" if verified==len(pin_rows) else "review_required"
    jwrite("input-pin-verification.json",{
      "schema":"numi.human.bilateral-forearm-preparation-1360.input-pin-verification.v1",
      "status":pin_status,"unique_pin_count":len(pin_rows),"verified_count":verified,
      "mismatch_count":sum(r["status"]=="mismatch" for r in pin_rows),
      "missing_count":sum(r["status"]=="missing" for r in pin_rows),
      "checked_sequentially":True,"sha256_algorithm":"SHA-256","pins":pin_rows})
    external=[]
    for r in pin_rows:
        if r["path"] in copied_sources: continue
        external.append({"path":r["path"],"sha256":r["actual_sha256"] or (r["expected_sha256"][0] if r["expected_sha256"] else None),
                         "declared_sha256":r["expected_sha256"],"size_bytes":r["size_bytes"],
                         "verified":r["status"]=="verified","reason":"Retained external immutable input; not copied"})
    jwrite("retained-external-files.json",{
      "evidence_boundary":"Large candidates, carrier and parent-term assets, composed payloads, accepted-pose packs, and the native receipt stay at their retained source paths.",
      "files":external})
    partial=[{"path":str(p),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(C2.rglob("*")) if p.is_file()]
    jwrite("proofs/composition013-002/status.json",{
      "status":"interrupted_incomplete_preflight","source_directory":str(C2),
      "available_artifact_count":len(partial),"available_artifacts":partial,
      "completed_preparation_json":False,"completed_composition_report":False,
      "note":"Only staged row patch files remain. This attempt is preserved as provenance and is not treated as a completed preflight; completed loader-preflight evidence is 013-003."})
    fail=F/"attempt-155-right-candidate150-carrier-assets-001.py"
    partial155=[{"path":str(p),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(S.rglob("*")) if p.is_file()]
    jwrite("proofs/bundle156/attempt155-provenance.json",{
      "status":"failed_first_packaging_attempt_assets_written_receipt_missing",
      "script_path":str(fail),"script_sha256":sha(fail),"script_size_bytes":fail.stat().st_size,
      "partial_output_directory":str(S),"partial_output_files":partial155,
      "reported_failure":"The first packager wrote carrier/geometry/terms and source deltas, then stopped while creating its provenance receipt because the pinned attempt095 evaluator source was absent from the input-hash map. Bundle156 finalization reused the saved outputs, added that source pin, and wrote the preparation receipt.",
      "completed_bundle_receipt":False})
    prevattr=jread(P/"anatomy-source-attribution.json")
    manifest=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
    route=F/"attempt-050/arrangement-001/diagnostic-061/fds-fdp-source-routes-061.json"
    jwrite("provenance/anatomy-source-attribution.json",{
      "schema":"numi.human.bilateral-forearm-preparation-1360.source-attribution.v1",
      "source_manifest_path":str(manifest),"source_manifest_sha256":sha(manifest),
      "source_payload_path":str(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),
      "source_payload_sha256":native["child_payload_sha256"],
      "source_archives":prevattr["source_archives"],
      "anatomy_attribution":prevattr["anatomy_attribution"],"anatomy_license":prevattr["anatomy_license"],
      "license_url":prevattr["license_url"],"source_url":prevattr["source_url"],
      "source_license_listing":prevattr["source_license_listing"],"derived_anatomy_license":prevattr["derived_anatomy_license"],
      "surfaces":[{"stable_id":131,"label":"right flexor digitorum superficialis"},
                  {"stable_id":133,"label":"right flexor digitorum profundus"}],
      "deformation_field":{"route_controls_path":str(route),"route_controls_sha256":sha(route),
       "interpretation":"Inferred passive visual deformation reference from authored MyoSim force-route controls; not measured tissue attachments and adds no physical route."},
      "changes":"Right FDS/FDP source-mesh conditioning, paired partition, and exact-route-field midpoint correction; preparation only."})
    source=jread(C150/"source-critical.json")
    f64=jread(C150/"current8000-f64-critical.json")
    f32s=jread(C150/"current8000-f32-separate-critical.json")
    f32f=jread(C150/"current8000-f32-fma-critical.json")
    chain=[
      (118,R118/"construction-checkpoint.json","proofs/bundle156/construction-chain/attempt-118-construction-checkpoint.json"),
      (121,R121/"construction-report.json","proofs/bundle156/construction-chain/attempt-121-construction-report.json"),
      (122,R122/"construction-report.json","proofs/bundle156/construction-chain/attempt-122-construction-report.json")]
    receipts=[(n,jread(src),dst,sha(src),str(src),src.stat().st_size) for n,src,dst in chain]
    counts=[len(r.get("accepted_contractions",[])) if isinstance(r.get("accepted_contractions"),list) else r.get("accepted_contractions",0) for _,r,_,_,_,_ in receipts]
    assert counts==[16,2,7],counts
    delta=jread(S/"construction-delta-104-to-122.json")
    assert [r[1]["candidate_sha256"] for r in receipts]==[
      "cfd81debd3317b076bddc82a6f58c5490748a02a100370d4958697c08775a2a2",
      "6f60f73521017f3e0f3e5b35e89dd78d29c88d29303a97f5f71950eb7e4548ce",
      "aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94"]
    assert receipts[1][1]["source_candidate118_sha256"]==receipts[0][1]["candidate_sha256"]
    assert receipts[2][1]["source_candidate118_sha256"]==receipts[1][1]["candidate_sha256"]
    rows={sid:{"face_incident_vertices_before":delta["rows"][sid]["candidate104_face_incident_vertices"],
      "face_incident_vertices_after":delta["rows"][sid]["candidate122_face_incident_vertices"],
      "face_incident_vertices_removed":delta["rows"][sid]["candidate104_face_incident_vertices"]-delta["rows"][sid]["candidate122_face_incident_vertices"],
      "listed_vertex_ids_before":delta["rows"][sid]["candidate104_listed_vertices"],
      "listed_vertex_ids_after":delta["rows"][sid]["candidate122_listed_vertices"],
      "listed_vertex_ids_removed":delta["rows"][sid]["active_vertex_ids_removed"],
      "listed_but_unreferenced_after":delta["rows"][sid]["candidate122_listed_unreferenced_vertices"],
      "oriented_face_count_before":delta["rows"][sid]["candidate104_oriented_face_count"],
      "oriented_face_count_after":delta["rows"][sid]["candidate122_oriented_face_count"],
      "oriented_faces_removed":delta["rows"][sid]["candidate104_oriented_face_count"]-delta["rows"][sid]["candidate122_oriented_face_count"],
      "all_common_coordinates_unchanged":delta["rows"][sid]["all_common_vertex_coordinates_unchanged"],
      "all_common_weights_unchanged":delta["rows"][sid]["all_common_vertex_weights_unchanged"]} for sid in ("131","133")}
    assert all(v["face_incident_vertices_removed"]==25 and
      v["listed_vertex_ids_before"]==v["listed_vertex_ids_after"] and v["listed_vertex_ids_removed"]==0 and
      v["listed_but_unreferenced_after"]==25 and v["oriented_faces_removed"]==50 for v in rows.values())
    correction={"schema":"numi.human.bilateral-forearm-preparation-1360.construction-chain-correction.v1",
      "scope":"Evidence interpretation overlay only; no candidate geometry or native input changed.",
      "supersedes":{"publication_file":"proofs/bundle156/construction-delta-104-to-122.json",
        "specific_correction":"accepted_contractions=7 is the final attempt122 stage count, not the cumulative104-to122 count."},
      "candidate_chain":{"reference_candidate104":{"path":str(F/"attempt-104-right-witness-band-refinement-001/result-001/candidate.pkl"),"sha256":"28ab429feb10615a5cfbd2345148d6b0619a7e75b41301b3f86f0a244202ad6d"},
        "attempt118_input_candidate116":{"path":str(F/"attempt-116-right-paired-cap-edge-flips-001/result-001/candidate.pkl"),"sha256":receipts[0][1]["input_hashes"][str(F/"attempt-116-right-paired-cap-edge-flips-001/result-001/candidate.pkl")],"role":"Preceding edge-flip conditioning input; not counted as a paired endpoint contraction."},
        "attempt118_output":{"sha256":receipts[0][1]["candidate_sha256"],"path":receipts[0][1]["candidate_path"]},
        "attempt121_output":{"sha256":receipts[1][1]["candidate_sha256"],"path":receipts[1][1]["candidate_path"]},
        "attempt122_output":{"sha256":receipts[2][1]["candidate_sha256"],"path":receipts[2][1]["candidate_path"]}},
      "construction_stages":[{"attempt":n,"accepted_contractions":counts[i],"report_path":src,"report_sha256":h,
        "publication_copy":dst,"candidate_sha256":r["candidate_sha256"],"gpu_launched":r.get("gpu_launched",False)}
        for i,(n,r,dst,h,src,size) in enumerate(receipts)],
      "cumulative_accepted_contractions":sum(counts),
      "rows":rows,
      "max_cumulative_endpoint_displacement_from_candidate104_um":delta["maximum_accepted_cumulative_displacement_from_candidate104_m"]*1e6,
      "interpretation":"Across stages118,121,122,25 paired-cap endpoint contractions were accepted (16+2+7). Candidate104-to122 face incidence decreases by25 vertices in each row and the oriented-face count decreases by50 in each row. The same25 global vertex IDs remain in each listed vertex inventory as unreferenced entries; active_vertex_ids_removed=0 describes the listed-ID inventory, not face incidence. Common retained coordinates and weights are unchanged. Tessellation changed, so exact surface coverage equivalence is not claimed.",
      "report_pins":[{"path":src,"sha256":h,"size_bytes":size,"publication_copy":dst} for n,r,dst,h,src,size in receipts]}
    jwrite("proofs/bundle156/construction-chain-correction-001.json",correction)
    correction_sha=sha(PUB/"proofs/bundle156/construction-chain-correction-001.json")
    creport=jread(C/"assets/report.json")
    rest=jread(C/"rest-registration-preflight.json")
    old=jread(P/"summary.json")
    comp_summary=jread(C/"preparation.json")
    summary={
      "status":"prepared_candidate_unadmitted_native013_not_run",
      "candidate":{"path":str(C150/"candidate.pkl"),"sha256":bundle["candidate"]["candidate150_sha256"],"admitted":False},
      "source_gate":{"rows":{k:v["self_count"] for k,v in source["rows"].items()},
                     "union_pairs":source["union"]["self_count"],"union_components":source["union"]["components"]},
      "current8000_float64":{"rows":{k:v["self_count"] for k,v in f64["rows"].items()},
          "union_pairs":f64["union"]["self_count"],"arithmetic":f64["arithmetic"]},
      "current8000_cpu_float32":{"separate":{"row_pairs":{k:v["self_count"] for k,v in f32s["rows"].items()},
          "row_degenerates":{k:v["degenerate_count"] for k,v in f32s["rows"].items()},"union_pairs":f32s["union"]["self_count"],"union_degeneracy":"not computed"},
          "fma":{"row_pairs":{k:v["self_count"] for k,v in f32f["rows"].items()},
          "row_degenerates":{k:v["degenerate_count"] for k,v in f32f["rows"].items()},"union_pairs":f32f["union"]["self_count"],"union_degeneracy":"not computed"}},
      "late151999_float64":{"rows":{k:v["self_count"] for k,v in late["rows"].items()},
          "union_pairs":late["union"]["self_count"],"union_components":late["union"]["components"]},
      "exact_route_replay":{"attempt153":"Float32-quantized-coordinate diagnostic; small nonzero deltas retained as superseded diagnostic.",
          "attempt154":"Exact rational source coordinates; zero changed full21 route-field scalar entries for candidate150."},
      "source_deltas":{"104_to_122":"25 paired-cap endpoint contractions across attempts118 (16), 121 (2), and 122 (7). The prior 7 count is the final122 stage only. From candidate104 to122, each row lost25 face-incident vertices and50 oriented faces; the same25 global IDs remain listed but unreferenced. Common retained coordinates/weights are unchanged; maximum cumulative endpoint displacement is21.4269 um. Tessellation changed; exact surface coverage is not claimed.",
          "122_to_150":"One exact midpoint vertex per row; common coordinates/weights unchanged; each row added4 and removed2 oriented faces."},
      "source_delta_correction":{"path":"proofs/bundle156/construction-chain-correction-001.json","sha256":correction_sha,"supersedes":"The interpretation of accepted_contractions in proofs/bundle156/construction-delta-104-to-122.json; its value7 describes attempt122 only."},
      "composition013_003":{"status":comp_summary["status"],"child_payload_sha256":comp_summary["child_payload_sha256"],
          "vertex_count":creport["vertex_count"],"index_count":creport["index_count"],
          "full_runtime_influences":comp_summary["full_runtime_influences"],"changed_stable_ids":[131,133],
          "binding_table_byte_exact":creport["binding_table_byte_exact"],"rest_gate":rest["status"],
          "max_registered_rest_error_um":{k:rest["rows"][k]["max_um"] for k in ("131","133")},
          "anatomical_admission_claimed":rest["anatomical_admission_claimed"]},
      "native013":{"status":native["status"],"gpu_run_performed":native["gpu_run_performed"],
          "anatomical_admission_claimed":native["anatomical_admission_claimed"],
          "immutable_input_count":native["immutable_input_count"],
          "preparation_sha256":sha(N/"preparation.json"),
          "run_declaration_sha256":sha(N/"baseline/run-declaration.json"),
          "child_payload_sha256":native["child_payload_sha256"],"extension_sha256":native["extension_sha256"],
          "not_run_reason":"The numi macOS desktop was logged out after the SSH drop; root requested user sign-in before native launch."},
      "prior_native012":{"only_prior_completed_native_forearm_evidence":True,"accepted_steps":old["accepted_steps"],
          "simulation_seconds":old["simulation_seconds"],"launcher_wall_seconds":old["launcher_wall_seconds"],
          "scope":"Prior left-only bounded native012 run; not bilateral native013 or long-run admission."},
      "unresolved":["No native013 execution or anatomical admission.",
          "CPU Float32 current8000 joined-union diagnostics retain7 pairs (separate accumulation) and4 pairs (FMA); union degeneracy was not computed.",
          "Pronator quadratus interfaces and six-part back remain unresolved."],
      "input_pin_verification_status":pin_status}
    jwrite("summary.json",summary)
    (PUB/"README.md").write_text(
f"""# Bilateral forearm candidate preparation (M6)

Candidate150 right FDS/FDP source geometry has exact-source and CPU Float64 clearance evidence at source, current step8000, and retained late step151999. Candidate150 remains unadmitted. At current step8000, CPU Float64 reports zero self and joined-union pairs. CPU Float32 diagnostics retain 7 joined-union pairs with separate accumulation and 4 with FMA; joined-union degeneracy was not computed. These are not native GPU results.

The exact attempt095 route-field replay at rational source coordinates reports zero changed full21 scalar coefficients for candidate150. Attempt153's Float32-coordinate query is retained as a quantization diagnostic and superseded for exact source evaluation by attempt154. The retained155 construction-delta field labeled 104-to-122 reports only the final attempt122 stage. The correction overlay in proofs/bundle156/construction-chain-correction-001.json pins attempts118,121,122 and establishes25 cumulative paired-cap endpoint contractions (16+2+7). Each row lost25 face-incident vertices and50 faces; those25 global IDs remain listed but are unreferenced. Common retained coordinates and weights remain unchanged, and no exact surface coverage is claimed. Large candidate, carrier, geometry arrays, and parent terms remain external and hash-pinned.

Composition013-003 is a loader preflight only. It prepared a 407,870-vertex, 2,203,746-index full-body payload and 842,394 runtime influences; stable rows131 and133 are the only replaced rows, and the binding table is byte-exact. Registered-rest gates passed. Native013 preparation and run declaration are retained with the launch command, but no native013 run occurred and no anatomical admission was made. The numi desktop was logged out after the SSH drop, and root requested sign-in before launch.

The only completed native forearm evidence cited here is the preceding left-only native012 run: 8,000 accepted steps, 16.00000076 simulation seconds, and {old["launcher_wall_seconds"]:.3f} seconds launcher wall. That bounded run does not qualify bilateral native013 or unresolved interfaces. Left/right pronator quadratus contacts and the six-part back remain open.

BodyParts3D attribution/license and the exact native013 surface manifest are recorded in provenance. Route-control provenance identifies authored MyoSim force-route references. The inferred passive deformation field does not establish tissue attachment truth or add a physical route.

The prepared native launch command is in execution/native013/launch-command.txt. It is included for review and was not executed. Root composition and audit scripts are in scripts/. Copied-file hashes and immutable input pins are in retained-files.json, retained-external-files.json, and input-pin-verification.json. The interrupted composition013-002 and failed first bundle pass155 remain separately recorded; completed preflight013-003 and finalized bundle156 supersede them without removing their provenance.
""")
    retained=list(copied)
    for rel in ["README.md","summary.json","input-pin-verification.json","retained-external-files.json",
      "proofs/composition013-002/status.json","proofs/bundle156/attempt155-provenance.json",
      "proofs/bundle156/construction-chain-correction-001.json",
      "provenance/anatomy-source-attribution.json","publish_bilateral_forearm_preparation_m6_1360.py"]:
        q=PUB/rel; digest=sha(q)
        retained.append({"archived_path":rel,"archived_sha256":digest,"source_path":str(q),
          "source_sha256":digest,"gzip":False,"size_bytes":q.stat().st_size,"copy_hash_match":True})
    jwrite("retained-files.json",retained)
    # Add preparation after its write to retained inventory and rewrite list.
    prep={
      "status":"prepared_candidate_unadmitted_native013_not_run",
      "scope":"Compact bilateral forearm evidence publication only; no candidate admission, native launch, commit, or push.",
      "candidate150_path":str(C150/"candidate.pkl"),"candidate150_sha256":bundle["candidate"]["candidate150_sha256"],
      "bundle156_preparation_path":str(B/"preparation.json"),"bundle156_preparation_sha256":sha(B/"preparation.json"),
      "composition013_003_path":str(C),"composition013_003_preparation_sha256":sha(C/"preparation.json"),
      "native013_path":str(N),"native013_preparation_sha256":sha(N/"preparation.json"),
      "native013_run_declaration_sha256":sha(N/"baseline/run-declaration.json"),
      "prior_native012_publication":str(P),
      "input_pin_verification_status":pin_status,"copied_file_count":len(copied)+1,
      "source_delta_correction_path":"proofs/bundle156/construction-chain-correction-001.json",
      "source_delta_correction_sha256":correction_sha,
      "external_file_count":len(external),"gpu_launched":False,"commit_or_push":False}
    jwrite("preparation.json",prep)
    item={"archived_path":"preparation.json","archived_sha256":sha(PUB/"preparation.json"),
      "source_path":str(PUB/"preparation.json"),"source_sha256":sha(PUB/"preparation.json"),
      "gzip":False,"size_bytes":(PUB/"preparation.json").stat().st_size,"copy_hash_match":True}
    retained.append(item)
    jwrite("retained-files.json",retained)
    print(json.dumps({"publication":str(PUB),"copied_file_count":len(copied),
      "external_file_count":len(external),"unique_pin_count":len(pin_rows),
      "verified_pins":verified,"pin_status":pin_status,
      "pin_mismatches":sum(r["status"]=="mismatch" for r in pin_rows),
      "pin_missing":sum(r["status"]=="missing" for r in pin_rows),
      "elapsed_seconds":round(time.monotonic()-t0,3)},indent=2))

if __name__=="__main__": main()
