#!/usr/bin/env python3
"""Compose four independently replayed comp0 kernel-fan batches onto current Fat."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492"
F=R/"run-038-kernel-fans-297"
BASE=R/"result-006/fat-comp0-conditioned-volume-proposal.npz"
BASEREPORT=R/"result-006/report.json"
LOCAL=F/"prep-extract-002/prepared/conditioned-comp0-local-input.npz"
VERIFY=[F/"verify-338"/f"batch-{i:03d}"/"receipt.json" for i in range(1,5)]
BATCHREPORTS=[F/f"batch-{i:03d}"/"report.json" for i in range(1,5)]
BATCHCANDIDATES=[F/f"batch-{i:03d}"/"kernel-fan-volume-proposal.npz" for i in range(1,5)]
EXPECTED={
 str(BASE):"54381f3d6072f95111010b3de0fdd7334f254f5466f5422b6936bf684083ee3e",
 str(BASEREPORT):"d9d63545e9081a7baeae10829c782f936691f2aefbf295de56568c925c4b16d7",
 str(LOCAL):"d4f1f7dc5d01d5f0dc147585a8ed6d6e2cb1f6f20e6804a84ff19d63c1abb8d5",
 str(BATCHREPORTS[0]):"6bf69c9b4f83338a9abeba57d545ae61fc117e0476f6c075e65496d9632e582d",
 str(BATCHREPORTS[1]):"cad40e48cc9dd197eed924ff566bb864757687490af49d59dd576be5ec0555bc",
 str(BATCHREPORTS[2]):"06700178b3d5e7c033156afd11f3040da1abf78f8ee0c59fc83eca500146628e",
 str(BATCHREPORTS[3]):"f11a2d7680caff6022a23d9aa9e91a49a22e7113bb83c034e5663a12311a151e",
 str(VERIFY[0]):"f787c8ac2e349f849b03a4e3d8e8c574a7dcc7b6bb28e6d45f45532f3a35d6c4",
 str(VERIFY[1]):"aecf5252ae71e75d7775c6b9ee3e22ec5eac56d293005007edec9364492987f9",
 str(VERIFY[2]):"6a1e0fd34f5b92550c5fe1428a9a1dc64cb4bb44e8158c3802d0eb6d86d4f5c0",
 str(VERIFY[3]):"0a650515eb9f89fcaee06467724e1a0b2491863c38a6b5732081101135b42e79",
}
CANDIDATE_SHA=[
 "039e358b1902df4460e4b432851c76ac430a6be00efb00731fce87e6c0bd7a11",
 "af884ce33e6da442cfe7c0418effece77f3e96865032657a91346d27e5dad203",
 "98e0b691370c65dbe6a470c59299516ee95193ed1585707837552b283daeeacd",
 "dbbfc172ec258e185fe7ce83cf598f1d9c91b43301b5c1f65b7f41bca277bc5e",
]
COUNTS=[178,150,131,28]
CORE=H/"fixed_boundary_volume_flip_198_root.py"
CAPS=H/"direct_connected_skin_caps_288_root.py"
CORE_SHA="650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
CAPS_SHA="4b7c46955fa286db38c6ee5ec1dc91ac427e97ed5184cbe964bc7934640d0791"
def sha(p):
    with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def loadnpz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def boundary_faces(t,pat):
    faces=np.asarray(t,np.int64)[:,pat].reshape(-1,3)
    sf=np.sort(faces,axis=1);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));ss=sf[order]
    starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];counts=np.diff(np.r_[starts,len(faces)])
    if np.any(counts>2):raise ValueError("nonmanifold face incidence in composed domain")
    signs=1-2*(((faces[:,0]>faces[:,1]).astype(np.int8)+(faces[:,0]>faces[:,2]).astype(np.int8)+(faces[:,1]>faces[:,2]).astype(np.int8))%2)
    summed=np.add.reduceat(signs[order],starts)
    if np.any(summed[counts==2]!=0):raise ValueError("same-direction internal face incidence")
    return faces[order[starts[counts==1]]]
def oriented_chain(faces):
    out={}
    for f in np.asarray(faces,np.int64).reshape(-1,3):
        a,b,c=map(int,f);key=tuple(sorted((a,b,c)))
        sgn=1 if ((a>b)+(a>c)+(b>c))%2==0 else -1
        out[key]=out.get(key,0)+sgn
    return {k:v for k,v in out.items() if v}
def chain_sha(chain):
    rows=np.asarray([(k[0],k[1],k[2],int(s)) for k,s in sorted(chain.items())],dtype="<i8")
    return hashlib.sha256(rows.tobytes()).hexdigest()
def edge_keys(t,npoints,patterns):
    e=np.sort(np.asarray(t,np.int64)[:,patterns].reshape(-1,2),axis=1)
    return np.unique(e[:,0].astype(np.int64)*int(npoints)+e[:,1].astype(np.int64))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    for p,w in EXPECTED.items():
        got=sha(Path(p))
        if got!=w:raise ValueError(f"pinned input hash mismatch: {p}: {got}")
    if sha(CORE)!=CORE_SHA or sha(CAPS)!=CAPS_SHA:raise ValueError("exact geometry owner source hash mismatch")
    started=time.monotonic()
    m=loadmod("fancompose198",CORE);caps=loadmod("fancompose288",CAPS)
    base=loadnpz(BASE);loc=loadnpz(LOCAL);parent_report=json.loads(BASEREPORT.read_text())
    if parent_report.get("status")!="PASS_EXACT_LOCAL_CONNECTIVITY_COMPOSITION_SOURCE_ONLY" or parent_report.get("physical_admission") is not False:
        raise ValueError("base is not a source-only pass")
    if parent_report.get("whole_candidate",{}).get("exact_sat",{}).get("status")!="INHERITED_FROM_EXACT_PARENT":
        raise ValueError("base exact SAT inheritance record absent")
    pp=np.asarray(base["points_world_m"],np.float32)
    if not np.array_equal(pp,base["points_world_m_f32"]):raise ValueError("base F32 point views differ")
    full_t=np.asarray(base["tetrahedra"],np.int64);full_m=np.asarray(base["material_indices"],np.int32)
    patch_rows=np.asarray(base["comp0_replacement_output_cell_rows"],np.int64)
    patch_t=np.asarray(base["comp0_replacement_output_tetrahedra_global_node_rows"],np.int64)
    local_p=np.asarray(loc["points_world_m"],np.float32)
    cur_t=np.asarray(loc["tetrahedra"],np.int64);cur_m=np.asarray(loc["material_indices"],np.int32)
    if not np.array_equal(local_p,pp) or not np.array_equal(full_t[patch_rows],cur_t) or not np.array_equal(full_m[patch_rows],cur_m):
        raise ValueError("local extraction does not exactly match base comp0 patch")
    if not np.array_equal(patch_t,cur_t) or not np.all(cur_m==16):raise ValueError("initial patch mesh/material mismatch")
    if len(patch_rows)!=13353:raise ValueError("unexpected initial patch row count")
    ao=np.asarray(base["comp0_conditioned_cell_ancestor_offsets"],np.int64)
    ar=np.asarray(base["comp0_conditioned_cell_ancestor_parent_output_cell_rows"],np.int64)
    if len(ao)!=len(patch_rows)+1 or ao[0]!=0 or ao[-1]!=len(ar):raise ValueError("initial ancestor CSR shape mismatch")
    ancestors=[set(map(int,ar[ao[i]:ao[i+1]])) for i in range(len(patch_rows))]
    if any(not x for x in ancestors):raise ValueError("empty initial comp0 cell ancestor union")
    cur_p=pp.copy()
    final_receipt=json.loads(VERIFY[3].read_text())
    if final_receipt.get("status")!="PASS_INDEPENDENT_CACHED_KERNEL_FAN_CHAIN_REPLAY" or final_receipt.get("operations")!=487:
        raise ValueError("final independent 338 chain receipt mismatch")
    if final_receipt.get("candidate_sha256")!=CANDIDATE_SHA[-1] or final_receipt.get("original_point_prefix_bit_identical") is not True:
        raise ValueError("final 338 candidate/prefix pins mismatch")
    payload_pins={x["path"]:x["sha256"] for x in final_receipt.get("case_payload_pins",[])}
    if len(payload_pins)!=487:raise ValueError(f"unexpected per-operation payload pin count: {len(payload_pins)}")
    batch_operation_counts=[];point_ancestor=[];operation_payload_pins=[]
    expected_input_sha=sha(LOCAL)
    for bi in range(4):
        vr=json.loads(VERIFY[bi].read_text());br=json.loads(BATCHREPORTS[bi].read_text())
        if vr.get("status")!="PASS_INDEPENDENT_CACHED_KERNEL_FAN_CHAIN_REPLAY" or vr.get("physical_admission") is not False:
            raise ValueError(f"independent 338 receipt {bi+1} not pass/source-only")
        if vr.get("input_sha256")!=sha(LOCAL) or br.get("input_sha256")!=expected_input_sha:
            raise ValueError(f"batch {bi+1} input chain mismatch")
        if vr.get("candidate_sha256")!=CANDIDATE_SHA[bi]:
            raise ValueError(f"batch {bi+1} independent 338 candidate pin mismatch")
        if sha(BATCHCANDIDATES[bi])!=CANDIDATE_SHA[bi] or br.get("output_sha256")!=CANDIDATE_SHA[bi]:
            raise ValueError(f"batch {bi+1} candidate hash mismatch")
        ops=br.get("operations",[])
        if len(ops)!=COUNTS[bi] or br.get("passing_operations")!=COUNTS[bi]:
            raise ValueError(f"batch {bi+1} operation count mismatch")
        opdirs=sorted((F/f"batch-{bi+1:03d}").glob("pass-*"),key=lambda q:int(q.name.split("-")[1]))
        if len(opdirs)!=COUNTS[bi]:raise ValueError(f"batch {bi+1} payload directory count mismatch")
        batch_input_t=cur_t
        batch_input_m=cur_m
        batch_input_anc=ancestors
        batch_removed=[]
        batch_added_t=[]
        batch_added_m=[]
        batch_added_anc=[]
        batch_points=cur_p
        removed_seen=set()
        for oi,(record,folder) in enumerate(zip(ops,opdirs)):
            rp=folder/"replacement.npz";rjson=folder/"report.json"
            expected_payload_sha=payload_pins.get(str(rp))
            if expected_payload_sha is None or sha(rp)!=expected_payload_sha:
                raise ValueError(f"unbound or changed operation payload {rp}")
            rec=json.loads(rjson.read_text())
            for k in ("old_cell_rows","new_cells","new_global_point_row","point_world_m","exact_F32_positive","exact_boundary_and_volume","exact_internal_face_crossings"):
                if k in record and record[k]!=rec.get(k):raise ValueError(f"batch report/pass record mismatch {k} {rp}")
            if rec.get("exact_F32_positive") is not True or rec.get("exact_boundary_and_volume") is not True or rec.get("exact_internal_face_crossings")!=0:
                raise ValueError(f"operation exact geometry receipt failed {rp}")
            delta=loadnpz(rp)
            old=np.asarray(delta["old_cell_rows"],np.int64);new_tets=np.asarray(delta["new_tetrahedra"],np.int64)
            newm=np.asarray(delta["material_indices"],np.int32);xyz=np.asarray(delta["new_point_world_m"],np.float32).reshape(3)
            if not np.array_equal(old,np.asarray(rec["old_cell_rows"],np.int64)) or len(np.unique(old))!=len(old):
                raise ValueError(f"operation old-cell map mismatch {rp}")
            if np.any(old<0) or np.any(old>=len(batch_input_t)):raise ValueError(f"operation old row outside batch input mesh {rp}")
            if removed_seen.intersection(map(int,old)):raise ValueError(f"batch operations overlap old input cells {rp}")
            removed_seen.update(map(int,old));batch_removed.extend(map(int,old))
            newrow=int(rec["new_global_point_row"])
            if newrow!=len(batch_points):raise ValueError(f"new point row is not append-only {rp}: {newrow} != {len(batch_points)}")
            if not np.array_equal(xyz,np.asarray(rec["point_world_m"],np.float32)):raise ValueError(f"new point coordinate mismatch {rp}")
            batch_points=np.vstack((batch_points,xyz.reshape(1,3))).astype(np.float32,copy=False)
            if np.any(new_tets<0) or np.any(new_tets>=len(batch_points)) or not np.any(new_tets==newrow):
                raise ValueError(f"new tets do not reference appended point {rp}")
            opmat=batch_input_m[old]
            if len(np.unique(opmat))!=1 or not np.all(newm==int(opmat[0])) or int(opmat[0])!=16:
                raise ValueError(f"operation material mismatch {rp}")
            op_anc=set().union(*(batch_input_anc[int(x)] for x in old))
            if not op_anc:raise ValueError(f"empty operation parent union {rp}")
            batch_added_t.append(new_tets);batch_added_m.append(newm)
            batch_added_anc.extend([op_anc.copy() for _ in range(len(new_tets))])
            point_ancestor.append(op_anc.copy())
            operation_payload_pins.append({"path":str(rp),"sha256":expected_payload_sha,"batch":bi+1,"pass":oi})
        removed=np.asarray(sorted(removed_seen),np.int64)
        keepmask=np.ones(len(batch_input_t),dtype=bool);keepmask[removed]=False;kept=np.flatnonzero(keepmask)
        cur_t=np.vstack([batch_input_t[kept],*batch_added_t])
        cur_m=np.concatenate([batch_input_m[kept],*batch_added_m])
        ancestors=[batch_input_anc[int(x)] for x in kept]+batch_added_anc
        cur_p=batch_points
        if len(cur_t)!=len(ancestors) or len(cur_t)!=len(cur_m):raise ValueError("batch replay shape mismatch")
        cand=loadnpz(BATCHCANDIDATES[bi])
        cand_p=np.asarray(cand["points_world_m"],np.float32)
        cand_t=np.asarray(cand["tetrahedra"],np.int64)
        cand_m=np.asarray(cand["material_indices"],np.int32)
        if not np.array_equal(cur_p,cand_p) or cur_t.shape!=cand_t.shape or cur_m.shape!=cand_m.shape:
            raise ValueError(f"batch {bi+1} replay shapes or point bytes differ from candidate")
        cur_keys=np.sort(cur_t,axis=1);cand_keys=np.sort(cand_t,axis=1)
        oc=np.lexsort((cur_keys[:,3],cur_keys[:,2],cur_keys[:,1],cur_keys[:,0]))
        on=np.lexsort((cand_keys[:,3],cand_keys[:,2],cand_keys[:,1],cand_keys[:,0]))
        if not np.array_equal(cur_keys[oc],cand_keys[on]):
            raise ValueError(f"batch {bi+1} canonical tetrahedron sets differ")
        if len(np.unique(cur_keys[oc],axis=0))!=len(cur_t) or len(np.unique(cand_keys[on],axis=0))!=len(cand_t):
            raise ValueError(f"batch {bi+1} has duplicate canonical cells")
        if not np.array_equal(cur_m[oc],cand_m[on]):
            raise ValueError(f"batch {bi+1} material mapping differs by exact cell key")
        # Exact canonical-key replay permits output row reordering while retaining complete ancestry.
        parity_cur=np.sum(cur_t[:, :, None]>cur_t[:, None, :],axis=(1,2))%2
        parity_cand=np.sum(cand_t[:, :, None]>cand_t[:, None, :],axis=(1,2))%2
        if not np.array_equal(parity_cur[oc]%2,parity_cand[on]%2):
            raise ValueError(f"batch {bi+1} changed tetrahedron orientation")
        rebound=[None]*len(cand_t)
        for k in range(len(cand_t)):
            rebound[int(on[k])]=ancestors[int(oc[k])]
        ancestors=rebound
        cur_p,cur_t,cur_m=cand_p,cand_t,cand_m
        expected_input_sha=CANDIDATE_SHA[bi]
        batch_operation_counts.append(len(ops))
    final_cand=loadnpz(BATCHCANDIDATES[-1])
    if len(cur_p)!=253712 or len(cur_t)!=15350 or len(cur_m)!=15350:raise ValueError("unexpected final local shape")
    if not np.array_equal(cur_p[:len(pp)],pp):raise ValueError("fan points changed original point prefix")
    if not np.array_equal(cur_t,final_cand["tetrahedra"]) or not np.array_equal(cur_p,final_cand["points_world_m"]):
        raise ValueError("replayed local mesh differs from final fan candidate")
    if not np.all(cur_m==16):raise ValueError("fan result changed local material")
    initial_boundary=oriented_chain(boundary_faces(np.asarray(loc["tetrahedra"],np.int64),m.PAT))
    final_local_boundary=boundary_faces(cur_t,m.PAT)
    if oriented_chain(final_local_boundary)!=initial_boundary or oriented_chain(final_local_boundary)!=oriented_chain(loc["desired_boundary_faces"]):
        raise ValueError("fan chain changed exact local oriented target shell")
    flux_before=caps.exact_volume_int(pp,np.asarray(loc["tetrahedra"],np.int64))
    flux_after=caps.exact_volume_int(cur_p,cur_t)
    if flux_before!=flux_after:raise ValueError("fan chain changed exact local signed volume")
    keep_mask=np.ones(len(full_t),dtype=bool);keep_mask[patch_rows]=False
    keep_rows=np.flatnonzero(keep_mask);outside_t=full_t[keep_rows]
    out_t=np.vstack((outside_t,cur_t));out_m=np.r_[full_m[keep_rows],cur_m];out_p=cur_p
    if len(out_t)!=len(full_t)-len(patch_rows)+len(cur_t):raise ValueError("global cell count mismatch")
    if len(out_p)!=253712:raise ValueError("global point count mismatch")
    patterns=np.asarray(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)),np.int64)
    outside_edges=edge_keys(outside_t,len(out_p),patterns)
    old_patch_edges=edge_keys(full_t[patch_rows],len(out_p),patterns)
    new_patch_edges=edge_keys(cur_t,len(out_p),patterns)
    fresh=np.setdiff1d(new_patch_edges,old_patch_edges,assume_unique=True)
    ix=np.searchsorted(outside_edges,fresh);valid=ix<len(outside_edges);hit=np.zeros(len(fresh),dtype=bool)
    hit[valid]=outside_edges[ix[valid]]==fresh[valid]
    if np.any(hit):raise ValueError("new comp0 edge also occurs in retained mesh: "+repr(fresh[np.flatnonzero(hit)[:8]].tolist()))
    old_boundary=boundary_faces(full_t,m.PAT);new_boundary=boundary_faces(out_t,m.PAT)
    old_chain=oriented_chain(old_boundary);new_chain=oriented_chain(new_boundary)
    target=np.asarray(base["proposed_current_desired_boundary_faces"],np.int64)
    if old_chain!=new_chain:raise ValueError("global oriented boundary changed from base")
    if new_chain!=oriented_chain(target):raise ValueError("composed boundary differs from pinned current target")
    canon=np.sort(out_t,axis=1);order=np.lexsort((canon[:,3],canon[:,2],canon[:,1],canon[:,0]))
    if np.any(np.all(canon[order[1:]]==canon[order[:-1]],axis=1)):raise ValueError("duplicate tetrahedra after fan composition")
    det,quality=m.measurements(out_p,out_t)
    if np.any(det<=0) or not np.all(out_m==16):raise ValueError("global F32 positive/material gate failed")
    local_det,local_q=m.measurements(out_p,cur_t)
    local_bad=int(np.count_nonzero(~m.passes(local_det,local_q)))
    if local_bad!=693:raise ValueError(f"final local quality count differs from 338: {local_bad}")
    global_bad=int(np.count_nonzero(~m.passes(det,quality)))
    old_global_bad=int(np.count_nonzero(~m.passes(*m.measurements(pp,full_t))))
    original_local_bad=int(np.count_nonzero(~m.passes(*m.measurements(pp,loc["tetrahedra"]))))
    if original_local_bad!=1476:raise ValueError("initial local quality count mismatch")
    old_to_new=np.full(len(full_t),-1,np.int64);old_to_new[keep_rows]=np.arange(len(keep_rows),dtype=np.int64)
    payload=dict(base);payload["points_world_m"]=out_p;payload["points_world_m_f32"]=out_p
    payload["tetrahedra"]=out_t;payload["material_indices"]=out_m;payload["boundary_faces"]=new_boundary
    payload["output_cell_to_input_cell"]=np.r_[np.asarray(base["output_cell_to_input_cell"],np.int64)[keep_rows],np.full(len(cur_t),-1,np.int64)]
    old_i2o=np.asarray(base["input_cell_to_output_cell"],np.int64);i2o=np.full_like(old_i2o,-1)
    validmap=old_i2o>=0;i2o[validmap]=old_to_new[old_i2o[validmap]];payload["input_cell_to_output_cell"]=i2o
    payload["output_node_to_input_node"]=np.r_[np.asarray(base["output_node_to_input_node"],np.int64),np.full(len(out_p)-len(pp),-1,np.int64)]
    payload["comp0_output_cell_to_parent_D480_rows"]=np.r_[np.asarray(base["comp0_output_cell_to_parent_D480_rows"],np.int64)[keep_rows],np.full(len(cur_t),-1,np.int64)]
    payload["comp0_output_cell_to_source480_rows"]=np.r_[np.asarray(base["comp0_output_cell_to_source480_rows"],np.int64)[keep_rows],np.full(len(cur_t),-1,np.int64)]
    patch_out_rows=np.arange(len(keep_rows),len(out_t),dtype=np.int64)
    payload["comp0_replacement_output_cell_rows"]=patch_out_rows
    payload["comp0_replacement_output_tetrahedra_global_node_rows"]=cur_t
    payload["comp0_replacement_output_material_indices"]=cur_m
    payload["comp0_added_global_point_rows"]=np.r_[np.asarray(base["comp0_added_global_point_rows"],np.int64),np.arange(len(pp),len(out_p),dtype=np.int64)]
    payload["comp0_quality_delta"]=np.asarray([int(np.asarray(base["comp0_quality_delta"]).reshape(-1)[0])+(local_bad-original_local_bad)],np.int64)
    anc_offsets=np.r_[0,np.cumsum([len(x) for x in ancestors])].astype(np.int64)
    anc_rows=np.asarray([r for group in ancestors for r in sorted(group)],np.int64)
    payload["comp0_conditioned_cell_ancestor_offsets"]=anc_offsets
    payload["comp0_conditioned_cell_ancestor_parent_output_cell_rows"]=anc_rows
    payload["comp0_kernel_fan_added_global_point_rows"]=np.arange(len(pp),len(out_p),dtype=np.int64)
    p_anc_offsets=np.r_[0,np.cumsum([len(x) for x in point_ancestor])].astype(np.int64)
    p_anc_rows=np.asarray([r for group in point_ancestor for r in sorted(group)],np.int64)
    payload["comp0_kernel_fan_point_ancestor_offsets"]=p_anc_offsets
    payload["comp0_kernel_fan_point_ancestor_parent_output_cell_rows"]=p_anc_rows
    payload["comp0_kernel_fan_batch_candidate_sha256"]=np.asarray(CANDIDATE_SHA,dtype="<U64")
    payload["comp0_kernel_fan_batch_report_sha256"]=np.asarray([sha(x) for x in BATCHREPORTS],dtype="<U64")
    payload["comp0_kernel_fan_verify338_receipt_sha256"]=np.asarray([sha(x) for x in VERIFY],dtype="<U64")
    payload["comp0_kernel_fan_operation_count"]=np.asarray([sum(batch_operation_counts)],np.int64)
    payload["comp0_conditioning_sat_basis"]=np.asarray(["inherited whole-parent exact SAT; independently replayed positive exact-volume/boundary-preserving kernel-fan operations; no second whole-mesh SAT run"])
    if len(anc_offsets)!=len(cur_t)+1 or anc_offsets[0]!=0 or anc_offsets[-1]!=len(anc_rows):raise ValueError("final cell ancestor CSR malformed")
    if len(p_anc_offsets)!=len(out_p)-len(pp)+1 or p_anc_offsets[-1]!=len(p_anc_rows):raise ValueError("fan point ancestor CSR malformed")
    if any(not x for x in ancestors) or any(not x for x in point_ancestor):raise ValueError("missing fan ancestry union")
    out.mkdir(parents=True,exist_ok=False);candidate=out/"fat-comp0-kernel-fan-composed-volume-proposal.npz"
    np.savez_compressed(candidate,**payload)
    with np.load(candidate,allow_pickle=False) as z:
        for k,want in (("points_world_m",out_p),("points_world_m_f32",out_p),("tetrahedra",out_t),("material_indices",out_m),("boundary_faces",new_boundary),("comp0_conditioned_cell_ancestor_offsets",anc_offsets),("comp0_conditioned_cell_ancestor_parent_output_cell_rows",anc_rows),("comp0_kernel_fan_point_ancestor_offsets",p_anc_offsets),("comp0_kernel_fan_point_ancestor_parent_output_cell_rows",p_anc_rows)):
            if not np.array_equal(z[k],want):raise ValueError("serialized candidate mismatch "+k)
    report={
      "schema":"numi.fat.comp0-connected-kernel-fan-global-composition.v1",
      "status":"PASS_EXACT_COMP0_KERNEL_FAN_GLOBAL_COMPOSITION_SOURCE_ONLY",
      "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
      "inputs":{"base_candidate":{"path":str(BASE),"sha256":sha(BASE)},"base_report":{"path":str(BASEREPORT),"sha256":sha(BASEREPORT)},"local_patch_input":{"path":str(LOCAL),"sha256":sha(LOCAL)},"independent_338_receipts":[{"path":str(p),"sha256":sha(p)} for p in VERIFY],"batch_reports":[{"path":str(p),"sha256":sha(p),"candidate_sha256":CANDIDATE_SHA[i]} for i,p in enumerate(BATCHREPORTS)],"exact_geometry_owner":{"path":str(CORE),"sha256":sha(CORE)},"exact_volume_owner":{"path":str(CAPS),"sha256":sha(CAPS)}},
      "operations":{"kernel_fan_operations":sum(batch_operation_counts),"batch_operation_counts":batch_operation_counts,"new_inferred_interior_points":len(point_ancestor),"first_new_point_row":int(len(pp)),"last_new_point_row":int(len(out_p)-1),"original_point_prefix_bit_identical":True,"exact_per_operation_positive_boundary_volume_replay":"PASS_INDEPENDENT_338"},
      "local_comp0":{"cells_before_fans":len(loc["tetrahedra"]),"cells_after_fans":len(cur_t),"quality_failures_before_fans":original_local_bad,"quality_failures_after_fans":local_bad,"quality_failure_reduction":original_local_bad-local_bad,"material":16,"boundary_faces":len(final_local_boundary),"boundary_unchanged":True,"oriented_boundary_sha256":chain_sha(oriented_chain(final_local_boundary)),"exact_signed_det6_before":str(flux_before),"exact_signed_det6_after":str(flux_after),"exact_volume_unchanged":True,"all_F32_determinants_positive":bool(np.all(local_det>0)),"per_operation_exact_positive_and_boundary_volume":"inherited from pinned independent 338 chain replay"},
      "whole_candidate":{"points":len(out_p),"cells_before":len(full_t),"cells_after":len(out_t),"global_quality_failures_before":old_global_bad,"global_quality_failures_after":global_bad,"all_F32_determinants_positive":bool(np.all(det>0)),"material_16":bool(np.all(out_m==16)),"duplicate_tets":0,"oriented_boundary_faces":len(new_boundary),"global_oriented_boundary_unchanged":True,"matches_current_target":True,"current_target_faces":len(target),"new_edge_retained_conflicts":0,"new_edges_checked":int(len(fresh)),"exact_sat":{"status":"INHERITED_WITH_EXACT_CAVITY_REPLAY","parent_candidate_sha256":sha(BASE),"parent_checked_pairs":int(np.asarray(base["comp0_exact_sat_checked_pairs"]).reshape(-1)[0]),"rerun_after_connectivity_changes":False,"basis":"all fan operations preserve their exact oriented local shell and exact signed volume; inherited parent exact SAT proof retained; no claim of second whole-mesh SAT scan"}},
      "lineage":{"conditioned_tet_to_parent_output_cell_map":"CSR comp0_conditioned_cell_ancestor_offsets / comp0_conditioned_cell_ancestor_parent_output_cell_rows; each replacement cell carries the complete union of retired cells' parent output rows","new_point_ancestry":"CSR comp0_kernel_fan_point_ancestor_offsets / comp0_kernel_fan_point_ancestor_parent_output_cell_rows","comp0_patch_source_union_D480_rows":int(len(base["comp0_conditioning_parent_union_D480_rows"])),"comp0_patch_source_union_source480_rows":int(len(base["comp0_conditioning_parent_union_source480_rows"])),"direct_single_parent_for_fan_tets":"not asserted; complete parent union carried"},
      "candidate":{"path":str(candidate),"sha256":sha(candidate),"size_bytes":candidate.stat().st_size},
      "limitations":"Source-only volume construction and quality conditioning evidence. Native quality remains incomplete; no mass mapping, native steps, physical admission, or physiological qualification.",
      "elapsed_s":time.monotonic()-started
    }
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()

