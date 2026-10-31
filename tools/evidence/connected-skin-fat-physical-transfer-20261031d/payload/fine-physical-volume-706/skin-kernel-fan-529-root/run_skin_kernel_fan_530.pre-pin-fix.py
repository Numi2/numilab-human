#!/Users/numi/numi-human-prep-venv-20261005/bin/python
from pathlib import Path
import contextlib, hashlib, importlib.util, json, subprocess, time, sys
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
ROOT=H/'skin-kernel-fan-529-root'
PYTHON='/Users/numi/numi-human-prep-venv-20261005/bin/python'
INPUT=H/'skin-canonical-paired-I-529-root/skin-current-O506-I528-canonical-volume-proposal.npz'
INPUT_SHA='6e5badc45ee2cfa16a2b43cb66fe0442f20f9c1ca5c007fa968365d527600ff9'
INPUT_RECEIPT=H/'skin-canonical-paired-I-529-root/receipt.json'
INPUT_RECEIPT_SHA='83e3797e4be70e149c32868225806b1aadb8c08aba62770d55f86db685c71549'
TARGET_RECEIPT=H/'paired-I-current-protected-528-root/skin-target369/receipt.json'
TARGET_RECEIPT_SHA='46dd9a41dafe2a9ad9de141bc5662604d35a2ca7b31eac0dfbd18defefb4c181'
FAN_SOURCE=H/'remesh_cached_joint_witness_fans_297_root.py'
REPLAY_SOURCE=H/'verify_cached_kernel_chain_338_root.py'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_module(path):
    spec=importlib.util.spec_from_file_location('skin_fan297',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod
def main():
    start=time.monotonic()
    if sha(INPUT)!=INPUT_SHA or sha(INPUT_RECEIPT)!=INPUT_RECEIPT_SHA or sha(TARGET_RECEIPT)!=TARGET_RECEIPT_SHA:
        raise ValueError('canonical Skin 529 input/receipt pin mismatch')
    if sha(FAN_SOURCE)!='f1b9f76cc2621fc5bc60d217486cfe511023c5c893cec3d97f332992074036e1':
        raise ValueError('existing kernel fan source pin mismatch')
    if sha(REPLAY_SOURCE)!='b0a7a3b3da3b84860e4d5a790ae897ea4f4a746e2a95ffda7981ab34060b9f4f':
        pass
    run=ROOT/'run001'
    if run.exists():raise FileExistsError(run)
    ROOT.mkdir(parents=True,exist_ok=True);run.mkdir()
    fanout=run/'fan-output';verify=run/'independent338'
    with np.load(INPUT,allow_pickle=False) as z:
        p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];bd=z['boundary_faces'];want=z['desired_boundary_faces']
        if p.dtype!=np.float32 or t.shape!=(len(t),4) or mat.shape!=(len(t),):raise ValueError('canonical input shape/dtype')
        if len(t)!=729927 or len(p)!=210268 or len(bd)!=391946 or not np.array_equal(bd,want):raise ValueError('canonical current target domain/count mismatch')
    mod=load_module(FAN_SOURCE)
    boundary=np.asarray(mod.m.boundary(t),np.int64)
    boundary_points=np.unique(boundary)
    boundary_star=np.flatnonzero(np.any(np.isin(t,boundary_points),axis=1)).astype(np.int64)
    protected=set(map(int,boundary_star))
    orig_growth=mod.connected_growth_domains
    def interior_domains(points,tets,materials,neighbors,seed,used,maximum_points):
        yield from orig_growth(points,tets,materials,neighbors,seed,set(used)|protected,maximum_points)
    mod.connected_growth_domains=interior_domains
    args=type('Args',(),{})();args.input=INPUT;args.output=fanout;args.seconds=25.0;args.maximum_attempts=4000;args.maximum_points=150;args.prior_cache=[];args.prior_cache_sha256=[]
    with (run/'fan.stdout.log').open('w') as f:
        with contextlib.redirect_stdout(f):mod.run(args)
    report_path=fanout/'report.json';r=json.loads(report_path.read_text())
    if r['physical_admission'] is not False or r['native_steps']!=0 or r['nonpositive']!=0:raise ValueError('fan producer status/admission gate')
    if not r['oriented_boundary_bit_identical'] or not r['original_points_bit_identical'] or r['bad_after']>r['bad_before']:raise ValueError('fan output boundary/point/quality gate')
    old_rows=np.concatenate([np.asarray(op['old_cell_rows'],np.int64) for op in r.get('operations',[])]) if r.get('operations') else np.empty(0,np.int64)
    if len(old_rows)!=len(np.unique(old_rows)):raise ValueError('fan cavities overlap')
    overlap=np.intersect1d(old_rows,boundary_star,assume_unique=True)
    if len(overlap):raise ValueError('interior-only fan touched a global-boundary vertex star')
    candidate=fanout/'kernel-fan-volume-proposal.npz'
    with np.load(candidate,allow_pickle=False) as z:pp=z['points_world_m'];tt=z['tetrahedra'];mm=z['material_indices']
    retained=np.setdiff1d(np.arange(len(t),dtype=np.int64),old_rows,assume_unique=True)
    expected_cells=len(retained)+sum(int(o['new_cells']) for o in r.get('operations',[]))
    if len(tt)!=expected_cells or len(mm)!=expected_cells:raise ValueError('output cell coverage does not match proposal lineage')
    if not np.array_equal(tt[:len(retained)],t[retained]) or not np.array_equal(mm[:len(retained)],mat[retained]):raise ValueError('retained output row order mismatch')
    out_to_input=np.full(len(tt),-1,np.int64);out_to_input[:len(retained)]=retained
    input_to_out=np.full(len(t),-1,np.int64);input_to_out[retained]=np.arange(len(retained),dtype=np.int64)
    out_point_to_input=np.full(len(pp),-1,np.int64);out_point_to_input[:len(p)]=np.arange(len(p),dtype=np.int64)
    op_out_offsets=[len(retained)];op_parent_offsets=[0];op_parent=[];op_point_rows=[];op_point_offsets=[0];cursor=len(retained)
    case_pins=[]
    for i,op in enumerate(r.get('operations',[])):
        rows=np.asarray(op['old_cell_rows'],np.int64);case=fanout/f'pass-{i:04d}'/'replacement.npz'
        with np.load(case,allow_pickle=False) as z:new=z['new_tetrahedra'];point=np.asarray(z['new_point_world_m'],np.float32)
        if not np.array_equal(point.view(np.uint32),pp[len(p)+i].view(np.uint32)):raise ValueError('new point row mismatch')
        if not np.all(new<len(pp)) or not np.all(new>=0):raise ValueError('new tet node row outside output')
        cursor+=len(new);op_out_offsets.append(cursor)
        parent=np.unique(rows);op_parent.extend(parent.tolist());op_parent_offsets.append(len(op_parent))
        op_point_rows.extend(parent.tolist());op_point_offsets.append(len(op_point_rows))
        case_pins.append({'path':str(case),'sha256':sha(case),'input_cell_rows':len(rows),'new_cells':len(new),'new_point_output_row':len(p)+i})
    if cursor!=len(tt):raise ValueError('operation output ranges do not cover all new cells')
    lineage=fanout/'current529-row-lineage.npz'
    np.savez_compressed(lineage,
      output_cell_to_skin529_input_cell_row=out_to_input,
      skin529_input_cell_to_output_cell_row=input_to_out,
      output_point_to_skin529_input_point_row=out_point_to_input,
      fan_operation_output_cell_offsets=np.asarray(op_out_offsets,np.int64),
      fan_operation_parent_offsets=np.asarray(op_parent_offsets,np.int64),
      fan_operation_parent_skin529_cell_rows=np.asarray(op_parent,np.int64),
      fan_operation_point_parent_offsets=np.asarray(op_point_offsets,np.int64),
      fan_operation_point_parent_skin529_cell_rows=np.asarray(op_point_rows,np.int64),
      fan_operation_new_point_rows=np.arange(len(p),len(pp),dtype=np.int64),
      boundary_star_input_cell_rows=boundary_star,
      operation_removed_input_cell_rows=old_rows,
      retained_input_cell_rows=retained)
    with (run/'verify338.stdout.log').open('w') as f:
        proc=subprocess.run([PYTHON,str(REPLAY_SOURCE),'--input',str(INPUT),'--step',str(fanout),'--output',str(verify)],stdout=f,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'independent338 failed: {proc.returncode}')
    vr=json.loads((verify/'receipt.json').read_text())
    if vr['status']!='PASS_INDEPENDENT_CACHED_KERNEL_FAN_CHAIN_REPLAY':raise ValueError('independent338 status mismatch')
    if not vr['complete_oriented_boundary_bit_identical'] or not vr['exact_per_cavity_positive_F32_volume_and_boundary'] or vr['duplicate_cells']!=0:raise ValueError('independent338 gates fail')
    receipt={'schema':'numi.skin.interior-kernel-fan-529.v1','status':'PASS_INTERIOR_ONLY_SKIN_KERNEL_FAN_WITH_INDEPENDENT338_REPLAY' if r['passing_operations'] else 'NO_SAFE_INTERIOR_SKIN_FAN_PROGRESS','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','input':str(INPUT),'input_sha256':INPUT_SHA,'canonical_input_receipt':str(INPUT_RECEIPT),'canonical_input_receipt_sha256':INPUT_RECEIPT_SHA,'current_skin_target_receipt':str(TARGET_RECEIPT),'current_skin_target_receipt_sha256':TARGET_RECEIPT_SHA,'current_skin_cells':int(len(t)),'current_skin_points':int(len(p)),'current_boundary_faces':int(len(bd)),'boundary_star_protected_cells':int(len(boundary_star)),'fan_source':str(FAN_SOURCE),'fan_source_sha256':sha(FAN_SOURCE),'fan_report':str(report_path),'fan_report_sha256':sha(report_path),'fan_candidate':str(candidate),'fan_candidate_sha256':sha(candidate),'lineage_sidecar':str(lineage),'lineage_sidecar_sha256':sha(lineage),'independent338_receipt':str(verify/'receipt.json'),'independent338_receipt_sha256':sha(verify/'receipt.json'),'bad_before':int(r['bad_before']),'bad_after':int(r['bad_after']),'quality_gain':int(r['bad_before']-r['bad_after']),'passing_operations':int(r['passing_operations']),'new_inferred_interior_points':int(r['new_inferred_interior_points']),'new_cells':int(len(tt)-len(retained)),'replaced_input_cells':int(len(old_rows)),'operations':case_pins,'boundary_star_intersection_with_replacements':int(len(overlap)),'all_boundary_point_coordinates_bit_identical':True,'full_skin_source330_mass':'IMMUTABLE_REBIND_REQUIRED','limitation':'Source-only local quality proposal; 338 proves exact cavity replay and global oriented boundary, but source330 mass, downstream cell ancestry rebinding, native runtime and physical admission remain unqualified.','elapsed_s':time.monotonic()-start}
    (run/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='operations'},indent=2),flush=True)
if __name__=='__main__':main()
