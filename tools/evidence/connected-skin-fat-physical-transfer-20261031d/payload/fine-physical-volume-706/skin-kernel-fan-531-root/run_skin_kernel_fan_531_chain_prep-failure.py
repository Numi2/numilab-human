#!/Users/numi/numi-human-prep-venv-20261005/bin/python
from pathlib import Path
import contextlib, hashlib, importlib.util, json, subprocess, time, sys
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
ROOT=H/'skin-kernel-fan-531-root'
PYTHON='/Users/numi/numi-human-prep-venv-20261005/bin/python'
BASE=H/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz'
INPUT=ROOT/'run002/fan-output/kernel-fan-volume-proposal.npz'
INPUT_SHA='0a4687eb31d30f706b1f82d9fcbd23080155a9bf0dea7b3d743cdb1e619c4873'
INPUT_RECEIPT=ROOT/'run002/receipt.json'
INPUT_RECEIPT_SHA='04948dfaf35151263da1580abb9ed87e5c353aa98f82a5c4d85f28104ea858ec'
PARENT_LINEAGE=ROOT/'run002/fan-output/current531-row-lineage.npz'
PARENT_LINEAGE_SHA='c5c649670d4555601d2bd5aa349d2bf2161f04263f0288f98005758febff318d'
TARGET_RECEIPT=H/'outer-O-connected-quality-530-root/candidate/qualification435001/skin-target437/receipt.json'
TARGET_RECEIPT_SHA='ea45e849d58cc495eb84130f2fb17c494c38f87d859157abf014574265f7882c'
FAN_SOURCE=H/'remesh_cached_joint_witness_fans_297_root.py'
REPLAY_SOURCE=H/'verify_cached_kernel_chain_338_root.py'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_module(path):
    spec=importlib.util.spec_from_file_location('skin_fan297_chain',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod
def csr_for_parent_input(parent, ncell, npoint):
    base_cell=np.asarray(parent['output_cell_to_skin531_input_cell_row'],np.int64)
    if len(base_cell)!=ncell:raise ValueError('parent cell map length mismatch')
    if 'output_cell_parent_offsets' in parent.files:
        cell_off=np.asarray(parent['output_cell_parent_offsets'],np.int64)
        cell_rows=np.asarray(parent['output_cell_parent_skin531_rows'],np.int64)
        if len(cell_off)!=ncell+1:raise ValueError('parent cell CSR offsets mismatch')
    else:
        op_out=np.asarray(parent['fan_operation_output_cell_offsets'],np.int64)
        op_po=np.asarray(parent['fan_operation_parent_offsets'],np.int64)
        op_pr=np.asarray(parent['fan_operation_parent_skin531_cell_rows'],np.int64)
        op_for=np.full(ncell,-1,np.int32)
        for j,(a,b) in enumerate(zip(op_out[:-1],op_out[1:])):op_for[int(a):int(b)]=j
        counts=np.ones(ncell,np.int64);counts[base_cell>=0]=1
        for row in np.flatnonzero(base_cell<0):
            j=int(op_for[row])
            if j<0:raise ValueError(f'parent synthetic cell has no operation lineage: {row}')
            counts[row]=op_po[j+1]-op_po[j]
        cell_off=np.concatenate([np.zeros(1,np.int64),np.cumsum(counts,dtype=np.int64)])
        cell_rows=np.empty(int(cell_off[-1]),np.int64)
        for row in range(ncell):
            lo,hi=int(cell_off[row]),int(cell_off[row+1])
            if base_cell[row]>=0:cell_rows[lo]=base_cell[row]
            else:
                j=int(op_for[row]); plo,phi=int(op_po[j]),int(op_po[j+1]);src=op_pr[plo:phi]
                if len(src)!=hi-lo:raise ValueError('parent per-operation ancestry size mismatch')
                cell_rows[lo:hi]=src
    if cell_off[0]!=0 or cell_off[-1]!=len(cell_rows) or np.any(cell_off[1:]<cell_off[:-1]):raise ValueError('parent cell CSR malformed')
    base_point=np.asarray(parent['output_point_to_skin531_input_point_row'],np.int64)
    if len(base_point)!=npoint:raise ValueError('parent point map length mismatch')
    if 'output_point_parent_offsets' in parent.files:
        point_off=np.asarray(parent['output_point_parent_offsets'],np.int64)
        point_rows=np.asarray(parent['output_point_parent_skin531_cell_rows'],np.int64)
    else:
        point_off=np.zeros(npoint+1,np.int64); point_map={}
        op_points=np.asarray(parent['fan_operation_new_point_rows'],np.int64)
        op_po=np.asarray(parent['fan_operation_point_parent_offsets'],np.int64)
        op_pr=np.asarray(parent['fan_operation_point_parent_skin531_cell_rows'],np.int64)
        for j,row in enumerate(op_points):point_map[int(row)]=j
        counts=np.zeros(npoint,np.int64)
        for row in np.flatnonzero(base_point<0):
            j=point_map.get(int(row))
            if j is None:raise ValueError(f'parent new point has no source lineage: {row}')
            counts[row]=op_po[j+1]-op_po[j]
        point_off=np.concatenate([np.zeros(1,np.int64),np.cumsum(counts,dtype=np.int64)])
        point_rows=np.empty(int(point_off[-1]),np.int64)
        for row,j in point_map.items():
            lo,hi=int(point_off[row]),int(point_off[row+1]); plo,phi=int(op_po[j]),int(op_po[j+1])
            if hi-lo!=phi-plo:raise ValueError('parent point ancestry size mismatch')
            point_rows[lo:hi]=op_pr[plo:phi]
    if len(point_off)!=npoint+1 or point_off[-1]!=len(point_rows):raise ValueError('parent point CSR malformed')
    return base_cell,cell_off,cell_rows,base_point,point_off,point_rows
def main():
    start=time.monotonic()
    if sha(BASE)!='aba0761c3690bd68d964a1aca472ef70385a70dad16eed75da34ccb01ba58ebe':raise ValueError('canonical531 source pin mismatch')
    if sha(INPUT)!=INPUT_SHA or sha(INPUT_RECEIPT)!=INPUT_RECEIPT_SHA or sha(PARENT_LINEAGE)!=PARENT_LINEAGE_SHA or sha(TARGET_RECEIPT)!=TARGET_RECEIPT_SHA:raise ValueError('round input/lineage/target pin mismatch')
    if sha(FAN_SOURCE)!='f1b9f76cc2621fc5bc60d217486cfe511023c5c893cec3d97f332992074036e1':raise ValueError('existing kernel fan source pin mismatch')
    if sha(REPLAY_SOURCE)!='cd125e2cdd301df65bdf21f0fe0fcd0d63a4330fbe03b844707763acb25bb563':raise ValueError('independent338 source pin mismatch')
    mod=load_module(FAN_SOURCE)
    run=ROOT/'run003'
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True)
    fanout=run/'fan-output';verify=run/'independent338'
    with np.load(INPUT,allow_pickle=False) as z:
        p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];bd=z['boundary_faces'];want=z['desired_boundary_faces']
        if p.dtype!=np.float32 or t.shape!=(len(t),4) or mat.shape!=(len(t),):raise ValueError('input shape/dtype')
        if len(t)!=729880 or len(p)!=210284 or len(bd)!=391918:raise ValueError('round input domain/count mismatch')
        if not np.array_equal(mod.m.boundary_identity(bd),mod.m.boundary_identity(want)):raise ValueError('input actual/desired oriented boundary mismatch')
    with np.load(PARENT_LINEAGE,allow_pickle=False) as z: parent={k:z[k] for k in z.files}
    prev_cell_base,prev_cell_off,prev_cell_rows,prev_point_base,prev_point_off,prev_point_rows=csr_for_parent_input(parent,len(t),len(p))
    boundary=np.asarray(mod.m.boundary(t),np.int64);boundary_points=np.unique(boundary)
    boundary_star=np.flatnonzero(np.any(np.isin(t,boundary_points),axis=1)).astype(np.int64)
    args=type('Args',(),{})();args.input=INPUT;args.output=fanout;args.seconds=25.0;args.maximum_attempts=4000;args.maximum_points=150;args.prior_cache=[];args.prior_cache_sha256=[]
    with (run/'fan.stdout.log').open('w') as f:
        with contextlib.redirect_stdout(f):mod.run(args)
    report_path=fanout/'report.json';r=json.loads(report_path.read_text())
    if r['physical_admission'] is not False or r['native_steps']!=0 or r['nonpositive']!=0:raise ValueError('fan producer admission/positivity gate')
    if not r['oriented_boundary_bit_identical'] or not r['original_points_bit_identical'] or r['bad_after']>r['bad_before']:raise ValueError('fan output boundary/point/quality gate')
    old_rows=np.concatenate([np.asarray(op['old_cell_rows'],np.int64) for op in r.get('operations',[])]) if r.get('operations') else np.empty(0,np.int64)
    if len(old_rows)!=len(np.unique(old_rows)):raise ValueError('fan cavities overlap')
    candidate=fanout/'kernel-fan-volume-proposal.npz'
    with np.load(candidate,allow_pickle=False) as z:pp=z['points_world_m'];tt=z['tetrahedra'];mm=z['material_indices']
    if pp.dtype!=np.float32 or not np.array_equal(pp[:len(p)].view(np.uint32),p.view(np.uint32)):raise ValueError('output point prefix moved')
    retained=np.setdiff1d(np.arange(len(t),dtype=np.int64),old_rows,assume_unique=True)
    expected_cells=len(retained)+sum(int(o['new_cells']) for o in r.get('operations',[]))
    if len(tt)!=expected_cells or len(mm)!=expected_cells:raise ValueError('output cell coverage mismatch')
    if not np.array_equal(tt[:len(retained)],t[retained]) or not np.array_equal(mm[:len(retained)],mat[retained]):raise ValueError('retained output row order mismatch')
    out_to_input=np.full(len(tt),-1,np.int64);out_to_input[:len(retained)]=retained
    input_to_out=np.full(len(t),-1,np.int64);input_to_out[retained]=np.arange(len(retained),dtype=np.int64)
    out_to_base=np.full(len(tt),-1,np.int64);out_to_base[:len(retained)]=prev_cell_base[retained]
    base_count=len(np.asarray(parent['skin531_input_cell_to_output_cell_row']))
    base_to_out=np.full(base_count,-1,np.int64)
    valid=out_to_base>=0
    if len(np.unique(out_to_base[valid]))!=int(valid.sum()):raise ValueError('duplicate canonical531 cell mapping')
    base_to_out[out_to_base[valid]]=np.flatnonzero(valid)
    out_parent_counts=np.empty(len(tt),np.int64)
    out_parent_counts[:len(retained)]=np.diff(prev_cell_off)[retained]
    op_out_offsets=[len(retained)];op_current_parent_offsets=[0];op_current_parent_rows=[];op_base_parent_offsets=[0];op_base_parent_rows=[];op_point_rows=[];op_point_offsets=[0];cursor=len(retained)
    case_pins=[]
    for i,op in enumerate(r.get('operations',[])):
        rows=np.asarray(op['old_cell_rows'],np.int64);case=fanout/f'pass-{i:04d}'/'replacement.npz'
        with np.load(case,allow_pickle=False) as z:new=z['new_tetrahedra'];point=np.asarray(z['new_point_world_m'],np.float32)
        newpointrow=len(p)+i
        if not np.array_equal(point.view(np.uint32),pp[newpointrow].view(np.uint32)):raise ValueError('new point row mismatch')
        if not np.all(new<len(pp)) or not np.all(new>=0):raise ValueError('new tet node row outside output')
        cursor+=len(new);op_out_offsets.append(cursor);out_parent_counts[cursor-len(new):cursor]=0
        base_union=np.unique(np.concatenate([prev_cell_rows[prev_cell_off[q]:prev_cell_off[q+1]] for q in rows]))
        if len(base_union)==0:raise ValueError('fan operation has empty transitive source parent union')
        op_current_parent_rows.extend(rows.tolist());op_current_parent_offsets.append(len(op_current_parent_rows))
        op_base_parent_rows.extend(base_union.tolist());op_base_parent_offsets.append(len(op_base_parent_rows))
        op_point_rows.extend(base_union.tolist());op_point_offsets.append(len(op_point_rows))
        out_parent_counts[cursor-len(new):cursor]=len(base_union)
        case_pins.append({'path':str(case),'sha256':sha(case),'input_cell_rows':len(rows),'source531_parent_rows':len(base_union),'new_cells':len(new),'new_point_output_row':newpointrow})
    if cursor!=len(tt):raise ValueError('operation output ranges do not cover all new cells')
    out_parent_offsets=np.concatenate([np.zeros(1,np.int64),np.cumsum(out_parent_counts,dtype=np.int64)])
    out_parent_rows=np.empty(int(out_parent_offsets[-1]),np.int64)
    for j,row in enumerate(retained):
        lo,hi=int(out_parent_offsets[j]),int(out_parent_offsets[j+1]);a,b=int(prev_cell_off[row]),int(prev_cell_off[row+1])
        if hi-lo!=b-a:raise ValueError('retained parent CSR count mismatch')
        out_parent_rows[lo:hi]=prev_cell_rows[a:b]
    cursor=len(retained)
    for i,op in enumerate(r.get('operations',[])):
        rows=np.asarray(op['old_cell_rows'],np.int64);nnew=int(op['new_cells']);lo=int(op_base_parent_offsets[i]);hi=int(op_base_parent_offsets[i+1]);parents=np.asarray(op_base_parent_rows[lo:hi],np.int64)
        for row in range(cursor,cursor+nnew):out_parent_rows[out_parent_offsets[row]:out_parent_offsets[row+1]]=parents
        cursor+=nnew
    out_point_to_input=np.full(len(pp),-1,np.int64);out_point_to_input[:len(p)]=np.arange(len(p),dtype=np.int64)
    out_point_to_base=np.full(len(pp),-1,np.int64);out_point_to_base[:len(p)]=prev_point_base
    point_counts=np.empty(len(pp),np.int64);point_counts[:len(p)]=np.diff(prev_point_off);point_counts[len(p):]=0
    cursorp=len(p)
    for i,op in enumerate(r.get('operations',[])):
        npoint=int(op['new_global_point_row'])
        if npoint!=cursorp:raise ValueError('fan point output ordering is not contiguous')
        a,b=op_base_parent_offsets[i],op_base_parent_offsets[i+1];point_counts[npoint]=b-a;cursorp+=1
    out_point_offsets=np.concatenate([np.zeros(1,np.int64),np.cumsum(point_counts,dtype=np.int64)])
    out_point_rows=np.empty(int(out_point_offsets[-1]),np.int64)
    for row in range(len(p)):
        lo,hi=int(out_point_offsets[row]),int(out_point_offsets[row+1]);a,b=int(prev_point_off[row]),int(prev_point_off[row+1])
        if hi-lo!=b-a:raise ValueError('retained point parent CSR count mismatch')
        out_point_rows[lo:hi]=prev_point_rows[a:b]
    for i,op in enumerate(r.get('operations',[])):
        row=int(op['new_global_point_row']);lo,hi=int(out_point_offsets[row]),int(out_point_offsets[row+1]);a,b=op_base_parent_offsets[i],op_base_parent_offsets[i+1]
        out_point_rows[lo:hi]=op_base_parent_rows[a:b]
    lineage=fanout/'current531-row-lineage.npz'
    np.savez_compressed(lineage,
      output_cell_to_current_input_cell_row=out_to_input,
      current_input_cell_to_output_cell_row=input_to_out,
      output_cell_to_skin531_input_cell_row=out_to_base,
      skin531_input_cell_to_output_cell_row=base_to_out,
      output_cell_parent_offsets=out_parent_offsets,
      output_cell_parent_skin531_rows=out_parent_rows,
      output_point_to_current_input_point_row=out_point_to_input,
      output_point_to_skin531_input_point_row=out_point_to_base,
      output_point_parent_offsets=out_point_offsets,
      output_point_parent_skin531_cell_rows=out_point_rows,
      fan_operation_output_cell_offsets=np.asarray(op_out_offsets,np.int64),
      fan_operation_parent_offsets=np.asarray(op_current_parent_offsets,np.int64),
      fan_operation_parent_current_input_cell_rows=np.asarray(op_current_parent_rows,np.int64),
      fan_operation_source_parent_offsets=np.asarray(op_base_parent_offsets,np.int64),
      fan_operation_parent_skin531_cell_rows=np.asarray(op_base_parent_rows,np.int64),
      fan_operation_point_parent_offsets=np.asarray(op_point_offsets,np.int64),
      fan_operation_point_parent_skin531_cell_rows=np.asarray(op_point_rows,np.int64),
      fan_operation_new_point_rows=np.asarray([o['new_global_point_row'] for o in r.get('operations',[])],np.int64),
      boundary_star_input_cell_rows=boundary_star,
      operation_removed_input_cell_rows=old_rows,
      retained_input_cell_rows=retained)
    point_views=run/'canonical531-point-coordinate-views.npz'
    np.savez_compressed(point_views,points_world_m=pp,points_world_m_f32=pp.copy())
    with np.load(BASE,allow_pickle=False) as z:
        typed_pins={}
        for key in ('typed_I_candidate_path','typed_I_candidate_sha256','typed_O_candidate_path','typed_O_candidate_sha256'):
            if key in z.files:typed_pins[key]=str(z[key].item())
    with (run/'verify338.stdout.log').open('w') as f:proc=subprocess.run([PYTHON,str(REPLAY_SOURCE),'--input',str(INPUT),'--step',str(fanout),'--output',str(verify)],stdout=f,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'independent338 failed: {proc.returncode}')
    vr=json.loads((verify/'receipt.json').read_text())
    if vr['status']!='PASS_INDEPENDENT_CACHED_KERNEL_FAN_CHAIN_REPLAY' or not vr['complete_oriented_boundary_bit_identical'] or not vr['exact_per_cavity_positive_F32_volume_and_boundary'] or vr['duplicate_cells']!=0:raise ValueError('independent338 gate failed')
    receipt={'schema':'numi.skin.kernel-fan-531-chain.v1','status':'PASS_SKIN_KERNEL_FAN_WITH_INDEPENDENT338_REPLAY' if r['passing_operations'] else 'NO_SAFE_SKIN_FAN_PROGRESS','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','canonical_base_path':str(BASE),'canonical_base_sha256':'aba0761c3690bd68d964a1aca472ef70385a70dad16eed75da34ccb01ba58ebe','input':str(INPUT),'input_sha256':INPUT_SHA,'input_receipt':str(INPUT_RECEIPT),'input_receipt_sha256':INPUT_RECEIPT_SHA,'parent_lineage':str(PARENT_LINEAGE),'parent_lineage_sha256':PARENT_LINEAGE_SHA,'current_skin_target_receipt':str(TARGET_RECEIPT),'current_skin_target_receipt_sha256':TARGET_RECEIPT_SHA,'typed_source_pins':typed_pins,'current_skin_cells':len(t),'current_skin_points':len(p),'current_boundary_faces':len(bd),'boundary_star_cells_in_input':len(boundary_star),'boundary_star_cells_touched_by_replacements':len(np.intersect1d(old_rows,boundary_star,assume_unique=True)),'fan_source':str(FAN_SOURCE),'fan_source_sha256':sha(FAN_SOURCE),'fan_report':str(report_path),'fan_report_sha256':sha(report_path),'fan_candidate':str(candidate),'fan_candidate_sha256':sha(candidate),'lineage_sidecar':str(lineage),'lineage_sidecar_sha256':sha(lineage),'point_views_sidecar':str(point_views),'point_views_sidecar_sha256':sha(point_views),'independent338_receipt':str(verify/'receipt.json'),'independent338_receipt_sha256':sha(verify/'receipt.json'),'bad_before':int(r['bad_before']),'bad_after':int(r['bad_after']),'quality_gain':int(r['bad_before']-r['bad_after']),'passing_operations':int(r['passing_operations']),'new_inferred_interior_points':int(r['new_inferred_interior_points']),'new_cells':int(len(tt)-len(retained)),'replaced_input_cells':int(len(old_rows)),'all_input_point_coordinates_bit_identical':True,'oriented_global_boundary_exact':True,'full_skin_source330_mass':'IMMUTABLE_REBIND_REQUIRED','limitation':'Source-only fan increment; 338 independently proves each cavity. Per-cell parent CSR is transitively composed to canonical531 input rows. Both output coordinate views are emitted and bit-identical. Source330 mass, native runtime and physical admission remain unqualified.','elapsed_s':time.monotonic()-start}
    (run/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
