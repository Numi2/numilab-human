#!/usr/bin/env python3
"""Independent readback/replay for an 884c4 fixed-boundary cavity batch."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
PRODUCER=HERE/"fixed_boundary_cavity_884c4_cow_batch_root.py"
PRODUCER_SHA="334c69a0b54f0682ca60df4d10812602561aee4e33177df5558dccc270c03a86"
COW=HERE.parent/"npz-cow-append-925-root"/"npz_cow_append_925.py"
COW_SHA="4799f33d73a37072498565b6ff9c20d181916f0158e253fe5c7e9056ba2c0270"

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()

def ref(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}

def checked_ref(value,label):
    if not isinstance(value,dict) or not value.get("path") or not value.get("sha256"):
        raise ValueError(label+" missing path/SHA")
    p=Path(value["path"]).resolve()
    if not p.is_file():raise FileNotFoundError(label+" missing file "+str(p))
    got=sha(p)
    if got!=str(value["sha256"]):raise ValueError(label+" SHA mismatch "+got)
    return p

def import_pinned(name,path,digest):
    p=Path(path).resolve()
    got=sha(p)
    if got!=digest:raise RuntimeError(name+" pinned source mismatch "+got)
    spec=importlib.util.spec_from_file_location(name,p)
    if not spec or not spec.loader:raise RuntimeError("cannot import "+str(p))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def same_array(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return (a.dtype==b.dtype and a.shape==b.shape and
            a.flags.f_contiguous==b.flags.f_contiguous and
            a.tobytes(order="A")==b.tobytes(order="A"))

def source_refs(st):
    refs=st.get("source_surfaces")
    if not isinstance(refs,dict) or set(refs)!={"O","I","D"}:
        raise ValueError("state lacks exact O/I/D source references")
    out={}
    for role in ("O","I","D"):
        p=checked_ref(refs[role],role+" source")
        out[role]={"path":str(p),"sha256":str(refs[role]["sha256"])}
    return out

def state_of(module,path):
    root,st=module.P._source_state(Path(path).resolve())
    if not isinstance(st,dict):raise ValueError("state owner returned malformed source state")
    return root,st

def native_counts(module,points,tets,materials):
    det,q,good=module.native(points,tets)
    good=np.asarray(good,bool);materials=np.asarray(materials)
    per={}
    for m in np.unique(materials):
        mask=materials==m
        per[str(int(m))]=int(np.count_nonzero(~good[mask]))
    return {"total":int(np.count_nonzero(~good)),"per_material":per,"det":det,"q":q,"good":good}

def verify(args):
    if sha(args.parent_state)!=args.parent_state_sha256:
        raise ValueError("parent state SHA mismatch")
    if sha(args.batch_report)!=args.batch_report_sha256:
        raise ValueError("batch report SHA mismatch")
    report=json.loads(args.batch_report.read_text())
    producer_ref=report.get("producer",{})
    if Path(producer_ref.get("path","")).resolve()!=PRODUCER or producer_ref.get("sha256")!=PRODUCER_SHA:
        raise ValueError("batch report does not bind the frozen 884c4 producer")
    if report.get("physical_admission") is not False or int(report.get("native_steps",-1))!=0:
        raise ValueError("batch report makes an invalid admission/native claim")
    if report.get("input_state")!={"path":str(args.parent_state.resolve()),"sha256":args.parent_state_sha256}:
        raise ValueError("batch report parent state pin differs from CLI parent")
    jref=report.get("journal");sref=report.get("candidate_state")
    if not jref or not sref:
        if report.get("status")=="PREPARED_NO_ADMISSIBLE_PATCHES" and not any(report.get("accepted_counts",{}).values()):
            return {"schema":"h706.884c4-independent-replay.v1","status":"PASS_NO_OPS_NO_GEOMETRY_CHANGE",
                    "parent_state":ref(args.parent_state),"batch_report":ref(args.batch_report),
                    "source_geometry_unchanged":True,"physical_admission":False,"native_steps":0}
        raise ValueError("batch has no serialized journal/candidate state; output was not emitted")
    journal_path=checked_ref(jref,"operation journal")
    candidate_state_path=checked_ref(sref,"candidate state")
    journal=json.loads(journal_path.read_text())
    if journal.get("parent_state")!=report["input_state"]:
        raise ValueError("journal parent state pin differs from batch report")
    if journal.get("physical_admission") is not False or int(journal.get("native_steps",-1))!=0:
        raise ValueError("journal makes an invalid admission/native claim")
    owner=import_pinned("fixed_boundary_batch_884c4",PRODUCER,PRODUCER_SHA)
    engine=owner.P883
    parent_root,parent_st=state_of(engine,args.parent_state)
    candidate_root,candidate_st=state_of(engine,candidate_state_path)
    if candidate_root.get("source_state")!={"path":str(args.parent_state.resolve()),"sha256":args.parent_state_sha256}:
        raise ValueError("candidate state does not name exact input parent state")
    parent_sources=source_refs(parent_st);candidate_sources=source_refs(candidate_st)
    if candidate_sources!=parent_sources:
        raise ValueError("candidate changed an O/I/D source path or SHA")
    # Hash-checked typed archives are the unchanged-source proof for this volume-only operation.
    for role,pin in parent_sources.items():
        checked_ref(pin,role+" unchanged typed source")
    ctx=engine.load_context(args.parent_state,args.output.parent,fixture=args.fixture,source_role="D")
    if report.get("status") not in ("PASS_BATCH_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS",):
        raise ValueError("batch status is not an emitted replayable candidate: "+str(report.get("status")))
    roles=list(journal.get("roles",[]))
    if set(roles)!=set(journal.get("operations",{})) or set(roles)!=set(report.get("volume_candidates",{})):
        raise ValueError("role lists differ across journal and batch report")
    if not roles:raise ValueError("empty operation batch has unexpected candidate outputs")
    cow_owner=report.get("cow_owner",{})
    if Path(cow_owner.get("path","")).resolve()!=COW or cow_owner.get("sha256")!=COW_SHA:
        raise ValueError("batch report does not bind the frozen COW writer")
    cow_roles=report.get("cow_writes",{})
    results={};candidate_pins={}
    for role in roles:
        if role not in ("Skin","Fat"):raise ValueError("unsupported volume role "+str(role))
        parent_ref=dict(parent_st[role+"_current"])
        parent_path=checked_ref(parent_ref,role+" current parent volume")
        cache=ctx["caches"][role]
        parent_arrays=cache["arrays"]
        volref=report["volume_candidates"][role]
        if journal["volume_candidates"].get(role)!=volref:
            raise ValueError(role+" journal/report volume pins differ")
        candidate_path=checked_ref(volref,role+" candidate volume")
        candidate_pins[role]={"path":str(candidate_path),"sha256":str(volref["sha256"])}
        crecord=candidate_st.get(role+"_current",{})
        if crecord.get("path")!=str(candidate_path) or crecord.get("sha256")!=volref["sha256"]:
            raise ValueError(role+" candidate state current-volume pin mismatch")
        cowref=cow_roles.get(role,{}).get("report")
        cow_path=checked_ref(cowref,role+" COW receipt")
        cow=json.loads(cow_path.read_text())
        if cow.get("status")!="PASS_VERIFIED_STANDARD_NPZ" or cow.get("apfs_clonefile_used") is not True or cow.get("full_copy_fallback") is not False:
            raise ValueError(role+" COW receipt failed clone/no-fallback gate")
        if cow.get("source_sha256")!=parent_ref["sha256"] or Path(cow.get("source_path","")).resolve()!=parent_path:
            raise ValueError(role+" COW receipt parent differs from qualified parent volume")
        if cow.get("output_sha256")!=volref["sha256"] or Path(cow.get("output_path","")).resolve()!=candidate_path:
            raise ValueError(role+" COW receipt output differs from candidate volume")
        n_unchanged=int(cow.get("unchanged_compressed_members_bitwise_verified",-1))
        n_replaced=len(cow.get("replaced_keys",[]));n_new=len(cow.get("new_keys",[]));n_deleted=len(cow.get("deleted_keys",[]))
        if (n_unchanged!=len(cow.get("unchanged_compressed_members",[])) or
            int(cow.get("readback_arrays_bitwise_verified",-1))!=int(cow.get("output_member_count",-2)) or
            int(cow.get("source_member_count",-2))!=n_unchanged+n_replaced+n_deleted or
            int(cow.get("output_member_count",-2))!=n_unchanged+n_replaced+n_new):
            raise ValueError(role+" COW receipt array/member counts are inconsistent")
        child=engine.load_npz(candidate_path)
        ops_raw=journal["operations"][role]
        if not isinstance(ops_raw,list) or not ops_raw:raise ValueError(role+" has no operations")
        ops=[]
        for i,jop in enumerate(ops_raw):
            rows=np.asarray(jop["cell_rows"],np.int64)
            if not np.array_equal(rows,np.asarray(jop["complete_parent_union"],np.int64)):
                raise ValueError(role+" operation "+str(i)+" parent union is incomplete")
            ops.append({"edge":tuple(map(int,jop["edge"])),"rows":rows,
                "verts":np.asarray(jop["vertices"],np.int64),"newt":np.asarray(jop["new_tets"],np.int64).reshape((-1,4)),
                "material":int(jop["material"]),"olddef":float(jop["old_deficit"]),"newdef":float(jop["new_deficit"])})
        cache_t=np.asarray(cache["tets"],np.int64);cache_mat=np.asarray(cache["mat"]);points=np.asarray(cache["points"],np.float32)
        retired=np.unique(np.concatenate([x["rows"] for x in ops])).astype(np.int64)
        keep_mask=np.ones(len(cache_t),bool);keep_mask[retired]=False
        retained=np.flatnonzero(keep_mask).astype(np.int64)
        if not np.array_equal(child["retained_input_cell_rows"],retained):
            raise ValueError(role+" retained row map differs from operation retirement union")
        expected_new=np.vstack([x["newt"] for x in ops])
        if not np.array_equal(child["replacement_tetrahedra"],expected_new):
            raise ValueError(role+" replacement tetrahedra differ from journal")
        if not np.array_equal(child["tetrahedra"],np.vstack((cache_t[retained],expected_new))):
            raise ValueError(role+" full output tetrahedra not reconstructed from parent and journal")
        expected_materials=np.concatenate([np.full(len(x["newt"]),x["material"],dtype=cache_mat.dtype) for x in ops])
        if not np.array_equal(child["material_indices"],np.r_[cache_mat[retained],expected_materials]):
            raise ValueError(role+" full material array not reconstructed from parent and journal")
        if not np.array_equal(child["material_indices"][:len(retained)],cache_mat[retained]):
            raise ValueError(role+" retained materials differ")
        if not np.array_equal(child["tetrahedra"][:len(retained)],cache_t[retained]):
            raise ValueError(role+" retained parent cells differ")
        if not same_array(child["points_world_m"],parent_arrays["points_world_m"]):
            raise ValueError(role+" volume coordinate bytes changed")
        for key,a in parent_arrays.items():
            if key not in child:raise ValueError(role+" output omitted parent array "+key)
            aa=np.asarray(a);bb=np.asarray(child[key])
            if (aa.ndim==2 and aa.shape[1]==3 and aa.dtype.kind=="f" and
                any(token in key.lower() for token in ("point","coordinate","world_m")) and not same_array(aa,bb)):
                raise ValueError(role+" coordinate array changed: "+key)
        if not same_array(child["desired_boundary_faces"],parent_arrays["desired_boundary_faces"]):
            raise ValueError(role+" desired target face array changed")
        if engine.chain(engine.T.m.boundary(child["tetrahedra"]))!=engine.chain(cache["target"]):
            raise ValueError(role+" exact full oriented target boundary differs")
        if len(np.unique(np.sort(child["tetrahedra"],axis=1),axis=0))!=len(child["tetrahedra"]):
            raise ValueError(role+" duplicate output tetrahedra")
        # Rebuild all operation parent CSR directly from the journal, not candidate metadata.
        parent_rows=[];parent_offsets=[0];op_cell_rows=[];op_cell_offsets=[0];op_tet_offsets=[0];edge_rows=[]
        for x in ops:
            rows=np.asarray(x["rows"],np.int64);op_cell_rows.extend(map(int,rows));op_cell_offsets.append(len(op_cell_rows))
            edge_rows.append(x["edge"])
            for _ in x["newt"]:
                parent_rows.extend(map(int,rows));parent_offsets.append(len(parent_rows))
            op_tet_offsets.append(op_tet_offsets[-1]+len(x["newt"]))
        parent_rows=np.asarray(parent_rows,np.int64);parent_offsets=np.asarray(parent_offsets,np.int64)
        if not np.array_equal(child["replacement_parent_offsets"],parent_offsets) or not np.array_equal(child["replacement_parent_cell_rows"],parent_rows):
            raise ValueError(role+" complete replacement parent CSR differs from journal")
        if not np.array_equal(child["fixed_boundary_cavity_operation_cell_rows"],np.asarray(op_cell_rows,np.int64)):
            raise ValueError(role+" operation cell-row CSR differs from journal")
        if not np.array_equal(child["fixed_boundary_cavity_operation_cell_offsets"],np.asarray(op_cell_offsets,np.int64)):
            raise ValueError(role+" operation cell offsets differ from journal")
        if not np.array_equal(child["fixed_boundary_cavity_operation_new_tet_offsets"],np.asarray(op_tet_offsets,np.int64)):
            raise ValueError(role+" operation tet offsets differ from journal")
        if not np.array_equal(child["fixed_boundary_cavity_operation_edge_rows"],np.asarray(edge_rows,np.int64)):
            raise ValueError(role+" operation edge rows differ from journal")
        oldbad_sum=newbad_sum=0;local=[]
        for i,(x,jop) in enumerate(zip(ops,ops_raw)):
            cavity=owner.E.cavity(cache,x["edge"])
            if not np.array_equal(cavity["rows"],x["rows"]) or not np.array_equal(cavity["verts"],x["verts"]):
                raise ValueError(role+" operation "+str(i)+" cavity does not rederive")
            if not np.array_equal(cavity["oldb"],np.asarray(jop["old_boundary"],np.int64)):
                raise ValueError(role+" operation "+str(i)+" old boundary differs")
            old_exact=owner.P883.exact_positive(points,cache_t[x["rows"]])
            new_exact=owner.P883.exact_positive(points,x["newt"])
            if not old_exact or not new_exact or min(old_exact)<=0 or min(new_exact)<=0:
                raise ValueError(role+" operation "+str(i)+" exact F32 determinant sign failure")
            oldsum=sum(old_exact);newsum=sum(new_exact)
            flux=int(owner.ENG.exact_boundary_flux(points[cavity["verts"]],cavity["lb"]))
            if oldsum!=newsum or oldsum!=flux or owner.P883.chain(owner.T.m.boundary(x["newt"]))!=owner.P883.chain(cavity["oldb"]):
                raise ValueError(role+" operation "+str(i)+" exact local volume/flux/boundary mismatch")
            if str(oldsum)!=str(jop["old_det6"]) or str(newsum)!=str(jop["new_det6"]) or str(newsum-oldsum)!=str(jop["exact_delta_det6"]):
                raise ValueError(role+" operation "+str(i)+" journal integer det6 differs from replay")
            olddet,oldq,oldgood=owner.P883.native(points,cache_t[x["rows"]])
            newdet,newq,newgood=owner.P883.native(points,x["newt"])
            ob=int(np.count_nonzero(~oldgood));nb=int(np.count_nonzero(~newgood))
            od=owner.P883.deficit(olddet,oldq);nd=owner.P883.deficit(newdet,newq)
            if ob!=int(jop["old_bad"]) or nb!=int(jop["new_bad"]):
                raise ValueError(role+" operation "+str(i)+" native bad counts differ from journal")
            if nb>ob or not(nb<ob or nd<od-max(1e-8,1e-6*od)):
                raise ValueError(role+" operation "+str(i)+" native count/deficit gate failed")
            oldbad_sum+=ob;newbad_sum+=nb
            local.append({"edge":list(x["edge"]),"cells":len(x["rows"]),"vertices":len(x["verts"]),
                "old_bad":ob,"new_bad":nb,"old_deficit":od,"new_deficit":nd,
                "old_det6":str(oldsum),"new_det6":str(newsum),"exact_delta_det6":str(newsum-oldsum),
                "exact_boundary_flux_det6":str(flux),"exact_positive_replacements":len(new_exact)})
        oldstats=native_counts(owner.P883,points,cache_t,cache_mat)
        newstats=native_counts(owner.P883,points,child["tetrahedra"],child["material_indices"])
        if newstats["total"]>oldstats["total"]:raise ValueError(role+" full native bad count regresses")
        for m,n in newstats["per_material"].items():
            if n>oldstats["per_material"].get(m,0):raise ValueError(role+" material "+m+" native bad count regresses")
        actual=np.asarray(owner.T.m.boundary(child["tetrahedra"]),np.int64)
        if owner.P883.chain(actual)!=owner.P883.chain(cache["target"]):
            raise ValueError(role+" full rebuilt boundary chain differs from target")
        results[role]={"status":"PASS_EXTERNAL_INDEPENDENT_REPLAY","input_cells":len(cache_t),"output_cells":len(child["tetrahedra"]),
            "input_points":len(points),"output_points":len(child["points_world_m"]),"operation_count":len(ops),
            "retired_cells":len(retired),"replacement_cells":len(expected_new),"retained_cells_bit_identical":True,
            "all_coordinate_arrays_fixed":True,"exact_target_boundary_preserved":True,"duplicate_tets":0,
            "exact_positive_replacement_tets":len(expected_new),"replacement_parent_CSR_complete":True,
            "local_exact_delta_det6":["0"]*len(ops),"local_replays":local,"full_bad_before":oldstats["total"],
            "full_bad_after":newstats["total"],"full_bad_by_material_before":oldstats["per_material"],
            "full_bad_by_material_after":newstats["per_material"],"full_native_nonregression":True,
            "cow_source_sha256":cow["source_sha256"],"cow_output_sha256":cow["output_sha256"],
            "cow_compressed_members_preserved":cow["unchanged_compressed_members_bitwise_verified"],
            "physical_admission":False,"native_steps":0}
    return {"schema":"h706.884c4-independent-replay.v1","status":"PASS_EXTERNAL_INDEPENDENT_REPLAY_PENDING_FULL_CONTEXT",
        "parent_state":ref(args.parent_state),"batch_report":ref(args.batch_report),"journal":ref(journal_path),
        "candidate_state":ref(candidate_state_path),"volume_candidates":candidate_pins,
        "source_roles_unchanged":parent_sources,"source_geometry_unchanged":True,
        "roles":results,"source_all185":"PENDING_EXTERNAL_CURRENT_CONTEXT","mass_mapping":"PENDING",
        "physical_admission":False,"native_steps":0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--parent-state-sha256",required=True)
    ap.add_argument("--batch-report",type=Path,required=True)
    ap.add_argument("--batch-report-sha256",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--fixture",action="store_true")
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError("fresh verifier output path required")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=verify(a)
    result.update({"verifier":ref(Path(__file__)),"fixture":bool(a.fixture)})
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    if result["status"] not in ("PASS_EXTERNAL_INDEPENDENT_REPLAY_PENDING_FULL_CONTEXT","PASS_NO_OPS_NO_GEOMETRY_CHANGE"):
        raise SystemExit(2)

if __name__=="__main__":main()
