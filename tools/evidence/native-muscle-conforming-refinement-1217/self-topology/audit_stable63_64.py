#!/usr/bin/env python3
import hashlib, importlib.util, json, mmap, os, resource, struct, sys, time
from pathlib import Path
import numpy as np

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

R=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/native-refinement-1217-attempt2")
RUN=R/"native-run"
CAND=R.parent/"compose-final-001/candidate"
TISS=CAND/"bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=CAND/"bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
REPORT=CAND/"report.json"
RECEIPT=CAND/"resting-anatomy-receipt.json"
AUDIT_ROOT=R.parent/"stable63-64-self-topology-audit-1217-attempt2"
OUT=AUDIT_ROOT/"result"
DECL=AUDIT_ROOT/"audit-declaration.json"
STEPS=(0,9983,20000)
EXPECTED_TISS="1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48"
EXPECTED_MAN="f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac"
EXPECTED_REPORT="67f4b635b7ddfbbfe8486cf9f0373a97e98928ac251b43ac6394bf962ca91954"
EXPECTED_RECEIPT="10ecac382b42e08b8ba296f361d419a0a7db1fb6e458448596e127acbea14d62"
NHA_SHA="1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
RUNNER=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/native-refinement-1217-attempt2/audit_skin_1217.py")
RUNNER_SHA="3c8b2a10a66788ff198c91e3eaa3a12f5e4c2380225b39be456d1d5e581a44dc"

PRED_ROOT=Path("/Users/n/numi-human-clearance-envelope-source-1215/src/numilab_human")
PRED_INIT=PRED_ROOT/"__init__.py"
PRED=PRED_ROOT/"cardiac_cavity_intersections.py"
CLEARANCE=PRED_ROOT/"common_atlas_skin_clearance.py"
GEOMETRY=PRED_ROOT/"cardiac_cavity_geometry.py"
CORE=Path("/Users/n/numi-human-resting-evidence-20261005/native-common-skin-combined-cycle-audit-908/audit_cycle.py")
VALIDATOR=Path("/Users/n/numi-human-resting-evidence-20261005/cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py")
PINNED={
 str(PRED_INIT):"e06e46ca88bde0b4c4a6235c6135dafdb2e75d27633776dba0098698a082cf42",
 str(PRED):"934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4",
 str(CLEARANCE):"ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f",
 str(GEOMETRY):"f6e98744dad9e23cd3b505efc02e7faa618fccce81d2ee08b07f948ef182fb72",
 str(CORE):"2eba147ca37ea80a3ed12c8dd725986d77bcd60c194077fae6ddfc5961d9dcde",
 str(VALIDATOR):"eb9e5c762cdbab1e9b3f98c63a1580a21acba54637b105c28e2bc0a39187ba17",
 str(RUNNER):RUNNER_SHA,
 str(TISS):EXPECTED_TISS, str(MAN):EXPECTED_MAN, str(REPORT):EXPECTED_REPORT, str(RECEIPT):EXPECTED_RECEIPT,
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load "+str(path))
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def load_predicates():
    pkg="_hip1217_exact_1215"
    spec=importlib.util.spec_from_file_location(pkg,PRED_INIT,submodule_search_locations=[str(PRED_ROOT)])
    mod=importlib.util.module_from_spec(spec); sys.modules[pkg]=mod; spec.loader.exec_module(mod)
    ci=__import__(pkg+".cardiac_cavity_intersections",fromlist=["*"])
    cl=__import__(pkg+".common_atlas_skin_clearance",fromlist=["*"])
    return ci,cl

def jsonable(v):
    if isinstance(v,dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray): return jsonable(v.tolist())
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    if isinstance(v,Path): return str(v)
    return v

def pack_row_vertices_faces(path, wanted):
    clearance=load_predicates()[1]
    pos,surfs,counts=clearance._pack_surfaces(path,set(wanted))
    return pos,surfs,counts

def source_rows():
    raw=TISS.read_bytes()
    magic,abi,nr,nb,nv,ni,fp,src=struct.unpack_from("<8s6I32s",raw,0)
    require(magic==b"NHTISS4\0" and abi==5 and nr==150 and nb==512 and nv==433151 and ni==1884573,"candidate NHTISS header mismatch")
    require(len(raw)==64+nr*32+nb*36+nv*56+ni*4,"candidate NHTISS byte-range mismatch")
    rec=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(nr,8)
    vo=64+nr*32+nb*36; io=vo+nv*56
    idx=np.frombuffer(raw,dtype="<u4",count=ni,offset=io)
    verts=np.ndarray((nv,6),dtype="<f4",buffer=raw,offset=vo,strides=(56,4))
    out={}
    for sid in (63,64):
        found=np.flatnonzero(rec[:,6]==sid)
        require(len(found)==1,f"candidate stable id {sid} ambiguous/missing")
        row=rec[int(found[0])]
        vf,vc,ii,ic=map(int,(row[2],row[3],row[4],row[5]))
        faces=idx[ii:ii+ic].reshape(-1,3)-vf
        xyz=np.asarray(verts[vf:vf+vc,:3]).copy()
        out[sid]={"row":row.tolist(),"vertices":xyz,"faces":faces.astype(np.int64),"vertex_count":vc,"face_count":len(faces)}
    return out

def collect_inputs(run,invocation):
    paths=set(Path(p) for p in PINNED)
    for p in (R/"run-declaration.json",R/"execution.json",RUN/"invocation.json",RUN/"run-metadata.json",RUN/"native.log"):
        paths.add(p)
    for step in STEPS:
        paths.add(RUN/"accepted-geometry"/f"step-{step}.mrvpack")
        paths.add(RUN/"accepted-geometry"/f"step-{step}.receipt.json")
    for p in invocation.get("asset_sha256",{}):
        paths.add(Path(p))
    return paths

def main():
    start=time.monotonic(); cpu0=time.process_time()
    require(not OUT.exists(),"audit result directory already exists")
    PINNED[str(Path(__file__).resolve())]=sha(Path(__file__))
    PINNED[str(DECL.resolve())]=sha(DECL)
    for p,w in PINNED.items(): require(Path(p).is_file() and sha(p)==w,f"pinned input mismatch: {p}")
    run_decl=json.loads((R/"run-declaration.json").read_text())
    execution=json.loads((R/"execution.json").read_text())
    require(execution.get("returncode")==0 and execution.get("changed_inputs")=={},"native execution did not close cleanly/input drift")
    require(execution.get("declaration_sha256")==sha(R/"run-declaration.json"),"native execution does not bind run declaration")
    inv_path=RUN/"invocation.json"; meta_path=RUN/"run-metadata.json"
    inv=json.loads(inv_path.read_text()); meta=json.loads(meta_path.read_text())
    require(meta.get("exit_code")==0 and meta.get("argv")==inv.get("argv") and meta.get("asset_sha256")==inv.get("asset_sha256"),"native metadata/invocation mismatch")
    require(meta.get("loaded_metal_runtime",{}).get("verified") is True,"native Metal runtime not verified")
    asset_map=inv.get("asset_sha256",{})
    require(asset_map.get(str(TISS))==EXPECTED_TISS,"actual invocation does not bind the final stable64 NHTISS")
    require(meta.get("source_files_changed_during_run") in (False,[],None),"native source drift recorded")
    require(json.loads(MAN.read_text())["payload"]["sha256"]==EXPECTED_TISS,"candidate manifest payload pin mismatch")
    comp=json.loads(REPORT.read_text())
    require(comp.get("payload_sha256")==EXPECTED_TISS and comp.get("manifest_sha256")==EXPECTED_MAN,"candidate composition report pins differ")
    candidate=source_rows()
    run_mod=load_file(RUNNER,"_runner1217_selfaudit")
    audit_runner=run_mod.load_runner()
    static=run_mod.validate_static_sources()
    ci,clearance=load_predicates()
    core=load_file(CORE,"_core908_selfaudit")
    validator=load_file(VALIDATOR,"_validator_selfaudit")
    before_paths=collect_inputs(RUN,inv)
    for p,w in asset_map.items():
        require(Path(p).is_file() and sha(p)==w,f"native invocation asset hash mismatch: {p}")
    inputs_before={str(p.resolve()):sha(p) for p in sorted(before_paths,key=str)}
    outputs=[]
    runmod=RUNNER
    for step in STEPS:
        pack=RUN/"accepted-geometry"/f"step-{step}.mrvpack"
        receipt_path=RUN/"accepted-geometry"/f"step-{step}.receipt.json"
        receipt=json.loads(receipt_path.read_text())
        require(receipt.get("accepted_step")==step and receipt.get("physical_endpoint")=="accepted" and receipt.get("surface_audit_endpoint")=="passed","accepted capture receipt rejected")
        require(abs(float(receipt.get("accepted_time_s"))-step*0.002)<2e-6,"capture time mismatch")
        accepted=audit_runner.verify_pack(step,pack,receipt_path,validator,NHA_SHA)
        require(accepted.get("accepted_step")==step,"existing MRVPACK receipt validator failed")
        positions,surfaces,pack_counts=pack_row_vertices_faces(pack,{(51005,63),(51005,64)})
        capture={"step":step,"time_s":step*0.002,"pack_path":str(pack),"pack_sha256":sha(pack),"receipt_path":str(receipt_path),"receipt_sha256":sha(receipt_path),"pack_counts":pack_counts,"surfaces":{}}
        for sid in (63,64):
            key=(51005,sid)
            require(key in surfaces,f"captured pack missing stable id {sid}")
            face_global=np.asarray(surfaces[key]["faces"],dtype=np.int64)
            ids=np.unique(face_global)
            src=candidate[sid]
            require(len(ids)==src["vertex_count"],f"stable {sid} captured unique vertex count differs from candidate")
            base=int(ids[0])
            expected_ids=np.arange(base,base+src["vertex_count"],dtype=np.int64)
            require(np.array_equal(ids,expected_ids),f"stable {sid} pack vertices are not contiguous in candidate-local order")
            local_faces=face_global-base
            require(np.array_equal(local_faces,src["faces"]),f"stable {sid} capture triangle rows/order differ from final NHTISS source")
            xyz=np.asarray(positions[base:base+src["vertex_count"]],dtype="<f4")
            require(np.isfinite(xyz).all(),"captured xyz has non-finite values")
            records,valid_rows,degenerate=core.exact_records(xyz,local_faces,ci)
            label=("right" if sid==63 else "left")+" vastus lateralis"
            helper_report=None; helper_error=None
            try:
                target=clearance._prepare_closed_clearance_target(xyz,local_faces,allow_nested_enclosure=(sid==63))
                helper_report=target["report"]
            except Exception as e:
                helper_error=str(e)
                helper_report=getattr(e,"target_report",None)
            if helper_report is not None:
                self_audit=helper_report.get("self_intersection_audit",{})
                pairs=self_audit.get("triangle_pairs",[])
                # The quotient retains source face order. Keep exact witnesses for any rejected pair.
                witnesses=[]
                for i,j in pairs:
                    tri_a=tuple(ci.float32_point_lattice_key(xyz[v]) for v in local_faces[int(i)])
                    tri_b=tuple(ci.float32_point_lattice_key(xyz[v]) for v in local_faces[int(j)])
                    pts=sorted(set(ci.triangle_intersection_points(tri_a,tri_b)))
                    witnesses.append({"face_rows_local":[int(i),int(j)],"face_rows_global_pack_index":[int(i),int(j)],"pack_vertex_ids":[face_global[int(i)].tolist(),face_global[int(j)].tolist()],"intersection_points_lattice":[[str(c) for c in p] for p in pts],"intersection_points_m":[[float(c/ci._FLOAT32_LATTICE_DENOMINATOR) for c in p] for p in pts]})
            else:
                self_audit={}; pairs=[]; witnesses=[]
            topology=(helper_report or {}).get("topology",{})
            zero_degenerate=(len(degenerate)==0)
            self_zero=(self_audit.get("count")==0)
            allclosed=(helper_report or {}).get("all_components_closed_oriented_unused_free") is True
            embedded=(helper_report or {}).get("embedded_closed_target") is True
            outer_ok=(sid==63 and zero_degenerate and self_zero and allclosed and embedded)
            surface_result={
                "label":label,"semantic":51005,"stable_id":sid,
                "candidate_local_vertex_count":src["vertex_count"],"candidate_local_face_count":src["face_count"],
                "captured_pack_vertex_base":base,"captured_unique_vertex_count":len(ids),
                "captured_face_count":len(local_faces),
                "captured_faces_match_candidate_local_faces_exactly":True,
                "captured_surface_body_id":int(surfaces[key]["body"]),
                "degenerate_face_count_exact_float32":len(degenerate),"degenerate_face_rows_local":[int(x) for x in degenerate],
                "exact_self_intersection_count":self_audit.get("count"),"exact_self_aabb_candidate_pairs":self_audit.get("aabb_candidate_pairs"),
                "allowed_shared_vertex_or_edge_pairs":self_audit.get("allowed_shared_vertex_or_edge_pairs"),
                "exact_self_pair_rows_local":pairs,"exact_self_witnesses":witnesses,
                "topology_and_outer_envelope":helper_report,
                "helper_error":helper_error,
                "all_components_closed_oriented_unused_free":allclosed,
                "embedded_closed_target":embedded,
                "outer_envelope_eligible_for_external_skin_clearance":bool(outer_ok) if sid==63 else None,
                "inside_semantics":(helper_report or {}).get("inside_semantics","unqualified"),
            }
            capture["surfaces"][str(sid)]=surface_result
        outputs.append(capture)
    after_paths=collect_inputs(RUN,inv)
    inputs_after={str(p.resolve()):sha(p) for p in sorted(after_paths,key=str)}
    unchanged=inputs_before==inputs_after
    require(unchanged,"one or more pinned inputs changed during audit")
    OUT.mkdir()
    elapsed=time.monotonic()-start; cpu=time.process_time()-cpu0
    report={
      "schema":"numi.human.conforming-muscle-stable63-64-self-topology-audit.v1",
      "status":"complete_read_only_exact_self_topology_and_stable63_outer_eligibility" if unchanged else "input_changed",
      "native_run":str(RUN),"attempt_root":str(R),
      "native_execution":execution,"native_invocation_sha256":sha(inv_path),"native_run_metadata_sha256":sha(meta_path),"native_log_sha256":sha(RUN/"native.log"),
      "native_terminal_step":20000,"capture_steps":list(STEPS),"capture_times_s":[x*0.002 for x in STEPS],
      "candidate":{"payload_path":str(TISS),"payload_sha256":EXPECTED_TISS,"manifest_path":str(MAN),"manifest_sha256":EXPECTED_MAN,"composition_report_path":str(REPORT),"composition_report_sha256":EXPECTED_REPORT,"receipt_path":str(RECEIPT),"receipt_sha256":EXPECTED_RECEIPT,"stable_ids":[63,64]},
      "surface_identity_from_candidate_manifest":[x for x in json.loads(MAN.read_text())["source"]["surfaces"] if x.get("stable_id") in (63,64)],
      "predicate_sources":{p:{"sha256":h} for p,h in PINNED.items()},
      "runner_static_inputs":static,
      "scope":"Exact Float32 native-captured stable63/64 NHTISS self-pair/topology audit and stable63 external-skin outer-envelope eligibility at only steps 0,9983,20000. Does not duplicate or claim the separate 859-target skin cross-audit, does not cover intermediate time, and does not qualify mechanics.",
      "audit_declaration_sha256":sha(DECL),"audit_script_sha256":sha(Path(__file__)),
      "input_hashes_before":inputs_before,"input_hashes_after":inputs_after,"inputs_unchanged":unchanged,
      "elapsed_wall_seconds":elapsed,"elapsed_process_cpu_seconds":cpu,"max_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "captures":outputs,
    }
    (OUT/"report.json").write_text(json.dumps(jsonable(report),indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"output":str(OUT/"report.json"),"sha256":sha(OUT/"report.json"),"elapsed_wall_s":elapsed,"elapsed_cpu_s":cpu,"inputs_unchanged":unchanged,"step_results":[{"step":c["step"],"s63":{"self":c["surfaces"]["63"]["exact_self_intersection_count"],"components":[{"component_id":x["component_id"],"face_count":x["face_count"]} for x in c["surfaces"]["63"]["topology_and_outer_envelope"].get("component_face_rows",[])] if c["surfaces"]["63"]["topology_and_outer_envelope"] else None,"outer_ok":c["surfaces"]["63"]["outer_envelope_eligible_for_external_skin_clearance"],"error":c["surfaces"]["63"]["helper_error"]},"s64":{"self":c["surfaces"]["64"]["exact_self_intersection_count"],"components":[{"component_id":x["component_id"],"face_count":x["face_count"]} for x in c["surfaces"]["64"]["topology_and_outer_envelope"].get("component_face_rows",[])] if c["surfaces"]["64"]["topology_and_outer_envelope"] else None,"error":c["surfaces"]["64"]["helper_error"]}} for c in outputs]},sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
