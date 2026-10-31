#!/usr/bin/env python3
"""Indexed379 exact volume replay for the pinned D500 raw contraction proposal.
This source-only transfer preserves the full point row domain and every retained
input cell row; it does not perform mass mapping or native admission.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse, hashlib, importlib.util, json, time
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
VOLUME=H/"fat-comp0-connected-warning-repair-412/run-042-current-54381-normalized-rebind/fat-current-typed-target-rebound-54381-lineage-corrected.npz"
SURFACE_INPUT=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/preflight-output/identity-seed-D.npz"
SURFACE_OUTPUT=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/surface-proposal-500um/candidate-D-short-edge-collapse-raw.npz"
SURFACE_REPORT=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/surface-proposal-500um/report.json"
SURFACE_LEDGER=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/surface-proposal-500um/accepted-operations.jsonl"
RUNCFG=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/preflight-output/run-config.json"
PREFLIGHT=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/preflight-output/preflight.json"
AUTHORIZED=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/preflight-output/allowed-D-edge-map.npz"
OWNER=H/"transfer_surface_contractions_indexed_379_root.py"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    if s is None or s.loader is None:raise RuntimeError("cannot import "+str(path))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def f32key(p):return np.asarray(p,np.float32).view(np.uint32).tobytes()

def signed_faces(faces):
    out=Counter()
    for f in np.asarray(faces,np.int64):
        a,b,c=map(int,f)
        inv=(a>b)+(a>c)+(b>c)
        out[(tuple(sorted((a,b,c))),1 if inv%2==0 else -1)]+=1
    return out

def closed_oriented_chain(faces):
    edges=Counter()
    for a,b,c in np.asarray(faces,np.int64):
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            key=(min(u,v),max(u,v));edges[key]+=1 if u<v else -1
    return {k:v for k,v in edges.items() if v}

def exact_det_sum(m,p32,tets):
    tets=np.asarray(tets,np.int64).reshape((-1,4))
    used=np.unique(tets)
    ip=m.integer_points(np.asarray(p32,np.float32)[used])
    ix={int(v):i for i,v in enumerate(used)}
    return sum(m.exact_det(ip,[ix[int(v)] for v in tet]) for tet in tets)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic()
    pin_paths=[VOLUME,SURFACE_INPUT,SURFACE_OUTPUT,SURFACE_REPORT,SURFACE_LEDGER,RUNCFG,PREFLIGHT,AUTHORIZED,OWNER]
    pins={str(p):sha(p) for p in pin_paths}
    vol=load(VOLUME);sin=load(SURFACE_INPUT);sout=load(SURFACE_OUTPUT)
    sr=json.loads(SURFACE_REPORT.read_text());pf=json.loads(PREFLIGHT.read_text());cfg=json.loads(RUNCFG.read_text())
    if sr.get("status")!="PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL":raise ValueError("raw surface batch is not PASS")
    if pf.get("status")!="PASS_D_SHORT_EDGE_PREFLIGHT_ONLY":raise ValueError("current D preflight is not PASS")
    if cfg.get("schema")!="h706.fixed-boundary-short-edge-collapse-run-config.v1":raise ValueError("run config schema mismatch")
    if vol["fat_current_target_rebind_schema"].item()!="numi.fat614.current-typed-I-D-wall-boundary-rebind.v1":raise ValueError("current target rebind schema mismatch")
    p64=np.asarray(vol["points_world_m"],np.float32);p32=np.asarray(vol["points_world_m_f32"],np.float32)
    if p64.shape!=p32.shape or not np.array_equal(p64,p32):raise ValueError("current volume F32 point views differ")
    t0arr=np.asarray(vol["tetrahedra"],np.int64);mat0=np.asarray(vol["material_indices"],np.int32)
    if t0arr.ndim!=2 or t0arr.shape[1]!=4 or len(mat0)!=len(t0arr):raise ValueError("current volume arrays malformed")
    if np.any(t0arr<0) or np.any(t0arr>=len(p32)):raise ValueError("current volume tet point index out of range")
    if not np.all(mat0==16):raise ValueError("current Fat volume contains non-16 material rows")
    if not np.array_equal(sin["points_world_m_f32"],sout["points_world_m_f32"]):raise ValueError("raw D output changed point coordinates")
    dpoint=np.asarray(vol["fat_current_target_rebind_D_point_to_volume_rows"],np.int64)
    if len(dpoint)!=len(sin["points_world_m_f32"]):raise ValueError("D-to-volume point map length mismatch")
    if np.any(dpoint<0) or np.any(dpoint>=len(p32)):raise ValueError("D-to-volume point map out of range")
    if not np.array_equal(p32[dpoint],np.asarray(sin["points_world_m_f32"],np.float32)):raise ValueError("D point coordinates do not map bitwise to current Fat volume")

    owner=module(OWNER,"indexed379_D500")
    m=owner.m
    p=np.asarray(p32,np.float32)
    t=t0arr.copy();mat=mat0.copy();active=np.ones(len(t),bool);node_map=np.arange(len(p),dtype=np.int64)
    base_boundary=owner.boundary(t)
    if "boundary_faces" in vol and not np.array_equal(owner.boundary_identity(base_boundary),owner.boundary_identity(np.asarray(vol["boundary_faces"],np.int64))):
        raise ValueError("recomputed current volume boundary differs from pinned candidate")
    boundary_keys={tuple(sorted(map(int,f))) for f in base_boundary}
    look={f32key(x):i for i,x in enumerate(p)}
    if len(look)!=len(p):raise ValueError("current Fat volume has duplicate F32 points")
    flat=t.reshape(-1);node_order=np.argsort(flat,kind="stable");counts=np.bincount(flat,minlength=len(p));node_offsets=np.empty(len(p)+1,np.int64);node_offsets[0]=0;np.cumsum(counts,out=node_offsets[1:])
    v={"p":p,"t":t,"mat":mat,"base_boundary":base_boundary,"boundary_keys":boundary_keys,"look":look,"active":active,"map":node_map,"accepted":[],"node_incidence_order":node_order,"node_incidence_offsets":node_offsets,"node_incidence_added":{}}

    # Reconstruct the exact preflight-authorized operation and volume-star mapping.
    amap=load(AUTHORIZED);drops=np.asarray(amap["D_allowed_collapsed_point_rows"],np.int64);keeps=np.asarray(amap["D_allowed_retained_point_rows"],np.int64);edge_offsets=np.asarray(amap["D_orientation_affected_cell_offsets"],np.int64);edge_rows=np.asarray(amap["D_orientation_affected_cell_rows"],np.int64);deltas=np.asarray(amap["D_orientation_exact_det6_delta_integer"]).astype(str);gains=np.asarray(amap["D_orientation_native_bad_cell_gain"],np.int64)
    if len(drops)!=len(keeps) or len(edge_offsets)!=len(drops)+1 or len(deltas)!=len(drops) or len(gains)!=len(drops):raise ValueError("authorized orientation arrays have inconsistent dimensions")
    auth={}
    for i,(drop,keep) in enumerate(zip(drops,keeps)):
        key=(int(drop),int(keep))
        if key in auth:raise ValueError("duplicate authorized orientation")
        auth[key]=i
    selected_ops=pf["preflight"]["selected_operations"]
    selected_by_orientation={}
    for e in selected_ops:
        for o in e["orientations"]:
            key=(int(o["drop_surface_row"]),int(o["keep_surface_row"]))
            if key in selected_by_orientation:raise ValueError("duplicate selected preflight orientation")
            selected_by_orientation[key]=o
    warns=set(map(int,pf["protection"]["run027_warning_volume_point_rows"]))
    raw_ops=[json.loads(line) for line in SURFACE_LEDGER.read_text().splitlines() if line.strip()]
    if len(raw_ops)!=int(sr["accepted_collapse_count"]):raise ValueError("raw surface operation count differs from report")
    sp=np.asarray(sin["points_world_m_f32"],np.float32);sf=np.asarray(sin["faces"],np.int64)
    if any(o.get("role")!="D" for o in raw_ops):raise ValueError("non-D operation in D batch")
    if len(sout["faces"])!=int(sr["outputs"]["D"]["output_faces"]):raise ValueError("raw surface face count differs from report")
    used_surface=set();accepted=[];rejected=[];affected_offsets=[0];affected_rows=[];all_oldgood=0
    exact_delta_total=0;expected_delta_total=0;expected_delta_match=0
    for oi,op in enumerate(raw_ops):
        drop=int(op["collapsed_input_point_row"]);keep=int(op["retained_input_point_row"]);key=(drop,keep)
        if key not in auth or key not in selected_by_orientation:raise ValueError(f"surface operation {oi} not in exact preflight authorization")
        ai=auth[key];po=selected_by_orientation[key]
        if drop in warns or keep in warns:raise ValueError(f"surface operation touches protected run027 warning point: {key}")
        if drop in used_surface or keep in used_surface:raise ValueError("raw surface operations are not vertex-disjoint")
        if len(op["retired_parent_face_rows"])==0:raise ValueError("operation has no retired source faces")
        if min(op["retired_parent_face_rows"])<0 or max(op["retired_parent_face_rows"])>=len(sf):raise ValueError("operation parent face row outside identity D domain")
        res,why=owner.try_op(v,sp,sf,op)
        if res is None:
            rejected.append({"surface_operation_index":oi,"patch_id":op["patch_id"],"drop_surface_row":drop,"keep_surface_row":keep,"reason":why})
            continue
        expected_rows=np.asarray(amap["D_orientation_affected_cell_rows"][edge_offsets[ai]:edge_offsets[ai+1]],np.int64)
        actual_rows=np.asarray(res["rows"],np.int64)
        if not np.array_equal(np.sort(expected_rows),np.sort(actual_rows)):
            raise ValueError(f"operation {oi} affected volume cell star differs from authenticated preflight map")
        dropv=int(look[f32key(sp[drop])]);keepv=int(look[f32key(sp[keep])])
        if int(po["drop_volume_row"])!=dropv or int(po["keep_volume_row"])!=keepv:raise ValueError("surface-to-volume operation point map differs from preflight")
        if not np.array_equal(np.sort(actual_rows),np.sort(np.asarray(po["affected_cell_rows"],np.int64))):raise ValueError("preflight selected orientation cell rows mismatch")
        if dropv in warns or keepv in warns:raise ValueError("transfer touches protected run027 warning point")
        de=exact_det_sum(m,p,res["new"])-exact_det_sum(m,p,res["old"])
        ex=int(deltas[ai]);expected_delta_total+=ex
        if de==ex:expected_delta_match+=1
        oldbad=int(res["bad_before"]);newbad=int(res["bad_after"])
        if oldbad-newbad!=int(gains[ai]):
            # Neighboring accepted stars may share volume cells; keep the measured sequential result.
            gain_match=False
        else:gain_match=True
        all_oldgood+=oldbad
        t[actual_rows]=res["new"];active[actual_rows[np.asarray(res["deleted"],bool)]]=False;node_map[int(res["retired_volume_node"])]=int(res["kept_volume_node"])
        exact_delta_total+=de;v["accepted"].append(oi);used_surface.update((drop,keep))
        # Track the complete changed volume star for independent replay and composition.
        affected_rows.extend(map(int,actual_rows));affected_offsets.append(len(affected_rows))
        accepted.append({"surface_operation_index":oi,"patch_id":op["patch_id"],"drop_surface_row":drop,"keep_surface_row":keep,"drop_volume_row":int(res["retired_volume_node"]),"keep_volume_row":int(res["kept_volume_node"]),"affected_input_cell_rows":actual_rows.tolist(),"deleted_input_cell_rows":actual_rows[np.asarray(res["deleted"],bool)].tolist(),"removed_cell_count":int(res["removed_cells"]),"bad_cells_before_local":oldbad,"bad_cells_after_local":newbad,"preflight_native_bad_cell_gain":int(gains[ai]),"sequential_gain_matches_preflight":gain_match,"exact_det6_delta_integer":str(de),"preflight_exact_det6_delta_integer":str(ex),"preflight_exact_delta_matches":bool(de==ex)})
    if not accepted:raise ValueError(f"no surface contractions transferred to Fat volume; rejected={len(rejected)}")

    active_rows=np.flatnonzero(active);out_t=t[active].copy();out_mat=mat[active].copy()
    if np.any(out_mat!=16):raise ValueError("material 16 changed during indexed transfer")
    det_before,q_before=m.measurements(p,t0arr);det_after,q_after=m.measurements(p,out_t)
    if np.any(det_after<=0):raise ValueError("nonpositive Float32 tet in transferred candidate")
    if np.any(det_before<=0):raise ValueError("pinned current input volume contains a nonpositive F32 tet")
    bad_before=int(np.count_nonzero(~m.passes(det_before,q_before)));bad_after=int(np.count_nonzero(~m.passes(det_after,q_after)))
    if bad_after>bad_before:raise ValueError("global volume quality defect count increased")
    actual_boundary=owner.boundary(out_t)
    mapped_boundary=np.asarray(node_map[base_boundary],np.int64)
    mapped_boundary=mapped_boundary[~np.any(mapped_boundary[:,1:]==mapped_boundary[:,:1],axis=1)&~np.any(mapped_boundary[:,2:]==mapped_boundary[:,:2],axis=1)&~np.any(mapped_boundary[:,2:]==mapped_boundary[:,1:2],axis=1)]
    if not np.array_equal(owner.boundary_identity(actual_boundary),owner.boundary_identity(mapped_boundary)):
        raise ValueError("transferred volume boundary differs from mapped current volume boundary chain")
    actual_chain=signed_faces(actual_boundary)
    if closed_oriented_chain(actual_boundary):raise ValueError("transferred volume boundary is not a closed oriented chain")

    # Rebuild the proposed current target by the same exact D vertex quotient.
    target=np.asarray(vol["desired_boundary_faces"],np.int64);tg=np.asarray(vol["desired_boundary_source_group"],np.int8);ts=np.asarray(vol["desired_boundary_source_rows"],np.int64);tf=np.asarray(vol["desired_boundary_source_face_rows"],np.int64)
    dparent=np.asarray(dpoint[ sf ],np.int64)
    target_lookup={}
    for j,tri in enumerate(target):target_lookup.setdefault(tuple(sorted(map(int,tri))),[]).append(j)
    dtarget_rows=[];orientation=Counter()
    for frow,tri in enumerate(dparent):
        key=tuple(sorted(map(int,tri)));hits=target_lookup.get(key,[])
        if len(hits)!=1:raise ValueError(f"D parent face {frow} maps to {len(hits)} current target faces")
        j=hits[0]
        if int(tg[j])!=1:raise ValueError("D current target face source-group code changed")
        a=list(map(int,tri));b=list(map(int,target[j]));perm=[a.index(x) for x in b];inv=sum(perm[x]>perm[y] for x in range(3) for y in range(x+1,3));sgn=1 if inv%2==0 else -1;orientation[sgn]+=1
        if sgn!=-1:raise ValueError("current Fat D target orientation is not uniformly reversed")
        dtarget_rows.append(j)
    if len(set(dtarget_rows))!=len(sf):raise ValueError("current D target face census is incomplete/nonunique")
    keep_target=np.ones(len(target),bool);keep_target[np.asarray(dtarget_rows,np.int64)]=False
    new_d_target=np.asarray(dpoint[np.asarray(sout["faces"],np.int64)],np.int64)[:,[0,2,1]]
    if len(np.unique(np.sort(new_d_target,axis=1),axis=0))!=len(new_d_target):raise ValueError("new D target contains duplicate unoriented faces")
    target_non_d=target[keep_target];target_updated=np.vstack((target_non_d,new_d_target))
    target_group=np.concatenate((tg[keep_target],np.ones(len(new_d_target),np.int8)))
    target_source_rows=np.concatenate((ts[keep_target],np.full(len(new_d_target),-1,np.int64)))
    target_source_faces=np.concatenate((tf[keep_target],np.full(len(new_d_target),-1,np.int64)))
    mapped_target=node_map[target]
    deg=np.any(mapped_target[:,0:1]==mapped_target[:,1:2],axis=1)|np.any(mapped_target[:,0:1]==mapped_target[:,2:3],axis=1)|np.any(mapped_target[:,1:2]==mapped_target[:,2:3],axis=1)
    mapped_target=mapped_target[~deg]
    if signed_faces(mapped_target)!=signed_faces(target_updated):raise ValueError("rebuilt target from raw D faces differs from exact mapped old target chain")
    if closed_oriented_chain(target_updated):raise ValueError("contracted current target is not a closed oriented chain")
    target_chain=signed_faces(target_updated);missing=sum((target_chain-actual_chain).values());extra=sum((actual_chain-target_chain).values())

    exact_denominator=1<<447
    exact_volume_delta=Fraction(exact_delta_total,6*exact_denominator)
    float_volume_before=float(np.sum(det_before,dtype=np.float64)/6.0)
    float_volume_after=float(np.sum(det_after,dtype=np.float64)/6.0)
    cell_to_out=np.full(len(t0arr),-1,np.int64);cell_to_out[active_rows]=np.arange(len(active_rows),dtype=np.int64)
    out_npz=out/"fat-D500-short-edge-quotient-volume-proposal.npz"
    np.savez_compressed(out_npz,points_world_m=np.asarray(vol["points_world_m"],np.float32),points_world_m_f32=p32,tetrahedra=out_t,material_indices=out_mat,boundary_faces=actual_boundary,retained_input_cell_rows=active_rows,input_cell_to_output_cell=cell_to_out,output_cell_to_input_cell=active_rows,input_node_to_output_node=node_map,output_node_to_input_node=np.arange(len(p),dtype=np.int64),mapped_current_boundary_faces=mapped_boundary,proposed_current_desired_boundary_faces=target_updated,proposed_current_desired_boundary_source_group=target_group,proposed_current_desired_boundary_source_rows=target_source_rows,proposed_current_desired_boundary_source_face_rows=target_source_faces,operation_cell_star_offsets=np.asarray(affected_offsets,np.int64),operation_cell_star_input_rows=np.asarray(affected_rows,np.int64),operation_surface_ledger_indices=np.asarray([x["surface_operation_index"] for x in accepted],np.int64),operation_retired_volume_point_rows=np.asarray([x["drop_volume_row"] for x in accepted],np.int64),operation_kept_volume_point_rows=np.asarray([x["keep_volume_row"] for x in accepted],np.int64),operation_exact_det6_delta_integer=np.asarray([x["exact_det6_delta_integer"] for x in accepted],dtype="U256"))
    out_sha=sha(out_npz)
    op_report={"schema":"h706.d500-indexed379-volume-transfer.v1","status":"PASS_D500_INDEXED379_VOLUME_QUOTIENT_PROPOSAL","input_volume":{"path":str(VOLUME),"sha256":pins[str(VOLUME)]},"surface_input":{"path":str(SURFACE_INPUT),"sha256":pins[str(SURFACE_INPUT)]},"raw_surface_candidate":{"path":str(SURFACE_OUTPUT),"sha256":pins[str(SURFACE_OUTPUT)]},"surface_report":{"path":str(SURFACE_REPORT),"sha256":pins[str(SURFACE_REPORT)]},"surface_operation_ledger":{"path":str(SURFACE_LEDGER),"sha256":pins[str(SURFACE_LEDGER)]},"authorized_preflight_map":{"path":str(AUTHORIZED),"sha256":pins[str(AUTHORIZED)]},"preflight_report":{"path":str(PREFLIGHT),"sha256":pins[str(PREFLIGHT)]},"run_config":{"path":str(RUNCFG),"sha256":pins[str(RUNCFG)]},"indexed379_owner":{"path":str(OWNER),"sha256":pins[str(OWNER)]},"fixed_boundary_exact_predicate_owner":{"path":str(Path(m.__file__)),"sha256":sha(Path(m.__file__))},"candidate":{"path":str(out_npz),"sha256":out_sha},"input_points":len(p),"output_points":len(p),"point_coordinates_bitwise_unchanged":True,"full_point_row_domain_preserved":True,"input_cells":len(t0arr),"output_cells":len(out_t),"removed_cell_count":len(t0arr)-len(out_t),"retained_input_cell_rows_count":len(active_rows),"all_material_indices_16":bool(np.all(out_mat==16)),"surface_operation_count":len(raw_ops),"transferred_operation_count":len(accepted),"rejected_volume_transfers":rejected,"transferred_operations":accepted,"operation_stars_match_authenticated_preflight":True,"protected_run027_warning_points_untouched":True,"preflight_exact_det6_delta_matches_count":expected_delta_match,"preflight_exact_det6_delta_mismatch_count":len(accepted)-expected_delta_match,"preflight_exact_det6_delta_sum_integer":str(expected_delta_total),"sequential_exact_det6_delta_sum_integer":str(exact_delta_total),"exact_det6_scale_denominator":str(exact_denominator),"exact_signed_volume_delta_m3":{"numerator":str(exact_volume_delta.numerator),"denominator":str(exact_volume_delta.denominator),"float":float(exact_volume_delta)},"binary64_total_volume_before_m3":float_volume_before,"binary64_total_volume_after_m3":float_volume_after,"binary64_total_volume_delta_m3":float_volume_after-float_volume_before,"all_output_float32_determinants_positive":bool(np.all(det_after>0)),"exact_positive_for_every_new_cell":True,"native_bad_cells_before":bad_before,"native_bad_cells_after":bad_after,"native_bad_cell_gain":bad_before-bad_after,"minimum_volume_before_m3":float(np.min(det_before/6)),"minimum_volume_after_m3":float(np.min(det_after/6)),"minimum_normalized_det_before":float(q_before.min()),"minimum_normalized_det_after":float(q_after.min()),"volume_boundary_equals_exact_mapped_input_boundary":True,"proposed_target_rebuilt_from_D_output_equals_mapped_old_target":True,"proposed_target_closed_oriented_chain":True,"volume_boundary_equals_proposed_current_target":bool(missing==0 and extra==0),"proposed_target_missing_oriented_face_records":int(missing),"proposed_target_extra_oriented_face_records":int(extra),"current_target_D_faces_mapped":len(dtarget_rows),"current_target_D_orientation":{"same":orientation[1],"reversed":orientation[-1]},"coordinates_changed":False,"source_coordinates_or_mass_changed":False,"mass_mapping":"NOT_RUN","native_steps":0,"physical_admission":False,"elapsed_seconds":time.monotonic()-t0,"limitations":["This is an indexed379 source-only D contraction volume proposal, not mass mapping or native installation.","The transformed current target boundary may retain the pre-existing incomplete mismatch; counts are reported explicitly.","A fresh typed 697 rebind and complete joint O/I/D+both-wall/all185 audit are separate gates."]}
    (out/"report.json").write_text(json.dumps(op_report,indent=2,sort_keys=True)+"\n")
    (out/"operation-volume-star-map.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in accepted))
    print(json.dumps({k:op_report[k] for k in ("status","candidate","transferred_operation_count","rejected_volume_transfers","output_cells","removed_cell_count","native_bad_cells_before","native_bad_cells_after","native_bad_cell_gain","exact_signed_volume_delta_m3","preflight_exact_det6_delta_matches_count","preflight_exact_det6_delta_mismatch_count","volume_boundary_equals_exact_mapped_input_boundary","proposed_target_missing_oriented_face_records","proposed_target_extra_oriented_face_records","elapsed_seconds")},indent=2))

if __name__=="__main__":main()
