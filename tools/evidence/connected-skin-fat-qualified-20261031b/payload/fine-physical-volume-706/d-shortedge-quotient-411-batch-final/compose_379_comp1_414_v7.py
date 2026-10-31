#!/usr/bin/env python3
"""Compose exact D411 endpoint quotient on top of the frozen disjoint comp1 patch."""
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
BASE_SHA='5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211'
TRANSFER=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/fat-D411-379-current406-volume-proposal.npz'
TRANSFER_SHA='1d1b2e722f150bd9c840cb1090de53bf6c7ad72cd046a019a1f33fc99f5a5824'
TRANSFER_REPORT=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/report.json'
TRANSFER_REPORT_SHA='e1009338bb3141ae38134c14a0bdf5bac6bc307de2f7cc8ae63e3934a5f089a4'
TRANSFER_INDEPENDENT=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/independent-413-v2/independent-verification.json'
TRANSFER_INDEPENDENT_SHA='5ed728315b30e1155364218f2f1fe5886c6f37017658a822ff09e14d196a9be6'
SIDECAR=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/operation-cell-star-map.npz'
SIDECAR_SHA='153d6fb0178a964b16c493233ef51df0e43b9e1925ed99c7231eb7a4bb7372b0'
COMP1=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/connected-fixed-surface-volume-proposal.npz'
COMP1_SHA='e11249bd297a7e08ae8d74fe406439289dfaf2f1f3069d1ebf83876e3d170a15'
COMP1_REPORT=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/report.json'
COMP1_REPORT_SHA='325cdd1b62907d1bc966f9518d9fba0a79e145df3599903b2359aba76dc480d4'
COMP1_INDEPENDENT=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/independent-full-audit-419.json'
COMP1_INDEPENDENT_SHA='40314094d4eb66505236e4ec712f1528dbd63b83d0af40469f47134e1e5da6dd'
COMP1_SHELL=H/'fat614-yy-volume-replay-004/run-037-D407-target-cavity-prep/connected-cavity-shells.npz'
COMP1_SHELL_SHA='d0b813675dbbf5fb7a8ef41b0b961b67f104de769d09afab3ae3e5c7aeb7e089'
FINAL_D=H/'fat-comp0-connected-warning-repair-412/run-007-final-D411-comp1-flips/candidate-D-three-warning-flips-typed.npz'
FINAL_D_SHA='5bcbe4c42618b84399edd9e878c192598efaa0c7c213b84b22ea85be1e40fd40'
FINAL_D_697=H/'fat-comp0-connected-warning-repair-412/run-007-final-D411-comp1-flips/actual-697-D411-comp1-flips-load-v1.json'
FINAL_D_697_SHA='2cadf9a0d4e9bb4df41f60815b1f4abcaba39c0083ddbd416694f6ff4c478a2f'
OWNER=H/'transfer_surface_contractions_indexed_379_root.py'
OWNER_SHA='ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37'
REG=H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py'
OUT=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001'
COMP0_EXTRA_CELL=799862

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def boundary_id(owner,fs):return np.asarray(owner.boundary_identity(np.asarray(fs,np.int64)),np.int64)
def counter_rows(x):return Counter(tuple(map(int,r)) for r in np.asarray(x,np.int64))
for p,h in ((BASE,BASE_SHA),(TRANSFER,TRANSFER_SHA),(TRANSFER_REPORT,TRANSFER_REPORT_SHA),(TRANSFER_INDEPENDENT,TRANSFER_INDEPENDENT_SHA),(SIDECAR,SIDECAR_SHA),(COMP1,COMP1_SHA),(COMP1_REPORT,COMP1_REPORT_SHA),(COMP1_INDEPENDENT,COMP1_INDEPENDENT_SHA),(COMP1_SHELL,COMP1_SHELL_SHA),(FINAL_D,FINAL_D_SHA),(FINAL_D_697,FINAL_D_697_SHA),(OWNER,OWNER_SHA)):
 if sha(p)!=h:raise ValueError(f'input pin changed: {p}: {sha(p)}')
owner=mod('indexed379_compose414',OWNER);m=owner.m;reg=mod('exact414',REG)
start=time.monotonic()
base=np.load(BASE,allow_pickle=False);tr=np.load(TRANSFER,allow_pickle=False);side=np.load(SIDECAR,allow_pickle=False);c=np.load(COMP1,allow_pickle=False);shell=np.load(COMP1_SHELL,allow_pickle=False);finald=np.load(FINAL_D,allow_pickle=False)
basep=np.asarray(base['points_world_m_f32'],np.float32);baset=np.asarray(base['tetrahedra'],np.int64);basem=np.asarray(base['material_indices'],np.int32)
N=len(basep); oldmap=np.asarray(tr['input_node_to_output_node'],np.int64)
if oldmap.shape!=(N,):raise ValueError('transfer node map does not span base406')
# run041 is on the current D407 rebind: exactly three appended D points and the entire base406 point prefix.
cp=np.asarray(c['points_world_m'],np.float32);ct=np.asarray(c['tetrahedra'],np.int64);cm=np.asarray(c['material_indices'],np.int32)
ret=np.asarray(c['retained_input_cell_rows'],np.int64)
if cp.shape!=(N+3,3) or not np.array_equal(cp[:N].view(np.uint32),basep.view(np.uint32)):raise ValueError('comp1 point prefix differs from base406')
if len(np.unique(ret))!=len(ret) or len(ret)!=len(baset)-2113:raise ValueError('unexpected comp1 retained cell count')
comp_rows=np.asarray(shell['input_cell_rows'],np.int64)
if len(comp_rows)!=2113 or len(np.unique(comp_rows))!=2113 or not np.array_equal(np.sort(np.setdiff1d(np.arange(len(baset),dtype=np.int64),ret)),np.sort(comp_rows)):
 raise ValueError('comp1 shell rows do not exactly explain the base-cell exclusion')
if not np.array_equal(ct[:len(ret)],baset[ret]) or not np.array_equal(cm[:len(ret)],basem[ret]):raise ValueError('run041 retained prefix is not exact input-cell retention')
replacement_start=len(ret);replacement=np.arange(replacement_start,len(ct),dtype=np.int64)
if not len(replacement):raise ValueError('comp1 has no replacement tets')
# Reconstruct the composed point quotient: 406 nodes follow D411; three D407 comp1 points stay identity.
fullmap=np.arange(len(cp),dtype=np.int64);fullmap[:N]=oldmap
mapped=fullmap[ct]
sort=np.sort(mapped,axis=1);deg=np.any(sort[:,1:]==sort[:,:-1],axis=1)
if np.any(deg[replacement]):raise ValueError('D411 quotient would degenerate a comp1 replacement tetrahedron')
if not np.array_equal(mapped[replacement],ct[replacement]):raise ValueError('D411 quotient would alter a comp1 replacement tetrahedron')
# Independently prove all run041 retained cells align with the indexed379 output by exact cell row map.
tr_ret=np.asarray(tr['retained_input_cell_rows'],np.int64)
inmap=np.full(len(baset),-1,np.int64);inmap[tr_ret]=np.arange(len(tr_ret),dtype=np.int64)
tout=np.asarray(tr['tetrahedra'],np.int64);tm=np.asarray(tr['material_indices'],np.int32)
retained_transfer_mask=inmap[ret]>=0
expected_rows=inmap[ret[retained_transfer_mask]]
if np.any(deg[:len(ret)][retained_transfer_mask]):raise ValueError('a surviving base cell unexpectedly degenerates under D411')
if np.any(~deg[:len(ret)][~retained_transfer_mask]):raise ValueError('a 379-deleted base cell was not degenerate after node quotient')
quot_ret=mapped[:len(ret)][retained_transfer_mask]
if not np.array_equal(quot_ret,tout[expected_rows]) or not np.array_equal(cm[:len(ret)][retained_transfer_mask],tm[expected_rows]):raise ValueError('exact retained-cell composition differs from indexed379 output')
# Build merged output in run041 order: quotient retained cells plus untouched replacement tets.
keep=~deg
outt=mapped[keep];outm=cm[keep]
outpoints=cp.copy();outp32=cp.copy()
if not np.array_equal(outt[:len(ret)],quot_ret):raise AssertionError('retained prefix changed during output assembly')
comp1_output_start=int(np.count_nonzero(keep[:replacement_start]))
comp1_rows_out=np.arange(comp1_output_start,len(outt),dtype=np.int64)
# The appended comp1 points and replacement cells must remain byte-identical.
if not np.array_equal(outt[comp1_rows_out],ct[replacement]) or not np.array_equal(outm[comp1_rows_out],cm[replacement]):raise AssertionError('comp1 replacement changed')
# Recompute the actual oriented boundary, and independently compare to quotient of run041 actual boundary.
actual=owner.boundary(outt)
oldactual=owner.boundary(ct);oldmapped=fullmap[oldactual];s=np.sort(oldmapped,axis=1);valid=~np.any(s[:,1:]==s[:,:-1],axis=1);expectedbd=oldmapped[valid]
if counter_rows(boundary_id(owner,actual))!=counter_rows(boundary_id(owner,expectedbd)):raise ValueError('composed actual boundary differs from quotient of run041 boundary')
# Preserve a clearly labelled preflip run041 target snapshot for diagnosis; it is not the final D411/flip target.
e112target=np.asarray(c['desired_boundary_faces'],np.int64);mapped_target=fullmap[e112target];s=np.sort(mapped_target,axis=1);target_valid=~np.any(s[:,1:]==s[:,:-1],axis=1);mapped_target=mapped_target[target_valid]
ac=counter_rows(boundary_id(owner,actual));tc=counter_rows(boundary_id(owner,mapped_target));missing=sum((tc-ac).values());extra=sum((ac-tc).values())
# Match comp1 row 797759 requested by comp0 owner and record whether D411 changes/drops it.
rowmap=np.full(len(baset),-1,np.int64)
kept_base_rows=ret[~deg[:len(ret)]]
rowmap[kept_base_rows]=np.arange(len(kept_base_rows),dtype=np.int64)
# Candidate output ordering is retained rows then replacement rows; all comp1 rows are after the retained block.
if COMP0_EXTRA_CELL>=len(baset):raise ValueError('comp0 witness cell outside base406')
comp0_output_row=int(inmap[COMP0_EXTRA_CELL])
comp0_in_transfer=np.asarray(tr['tetrahedra'][comp0_output_row],np.int64) if comp0_output_row>=0 else np.asarray([],np.int64)
comp0_e112_position=int(np.searchsorted(ret,COMP0_EXTRA_CELL)) if np.array_equal(ret,np.sort(ret)) and COMP0_EXTRA_CELL in set(map(int,ret)) else (int(np.flatnonzero(ret==COMP0_EXTRA_CELL)[0]) if COMP0_EXTRA_CELL in set(map(int,ret)) else -1)
comp0_positions=np.flatnonzero(ret==COMP0_EXTRA_CELL)
comp0_pos=int(comp0_positions[0]) if len(comp0_positions) else -1
comp0_merged_row=(int(np.count_nonzero(~deg[:comp0_pos])) if comp0_pos>=0 and not deg[comp0_pos] else -1)
comp0_was_changed=(not np.array_equal(comp0_in_transfer,baset[COMP0_EXTRA_CELL])) if comp0_output_row>=0 else None
# Count final all-volume normalized quality and check exact positivity for every changed D411 tet plus comp1 patch tets.
newdet,newq=m.measurements(outp32,outt);newgood=m.passes(newdet,newq)
exact_changed_rows=np.asarray(side['operation_changed_cell_rows'],np.int64)
exact_base_rows=np.unique(exact_changed_rows)
exact_kept_rows=exact_base_rows[rowmap[exact_base_rows]>=0]
exact_output_rows=rowmap[exact_kept_rows]
exact_changed=reg.exact_star_determinants(outp32,outt[exact_output_rows])
exact_comp1=reg.exact_star_determinants(outp32,outt[comp1_rows_out])
if not all(v>0 for v in exact_changed) or not all(v>0 for v in exact_comp1):raise ValueError('changed quotient or comp1 replacement contains exact nonpositive tet')
# Old-good regressions apply only to retained original rows. Removed or remeshed input rows are separately tallied.
old_det,old_q=m.measurements(basep,baset);oldgood=m.passes(old_det,old_q)
regress=kept_base_rows[oldgood[kept_base_rows] & ~newgood[:len(kept_base_rows)]]
comp1_replaced_old_good=int(np.count_nonzero(oldgood[comp_rows]))
d411_deleted_old_good=int(np.count_nonzero(oldgood[inmap<0]))
if len(regress):raise ValueError('retained original good cells regressed after composition')
# Base-cell map back to merged output, and new-cell rows use -1 in input-domain map.
input_to_output=np.full(len(baset),-1,np.int64);input_to_output[kept_base_rows]=rowmap[kept_base_rows]
output_to_input=np.r_[kept_base_rows, np.full(len(replacement),-1,np.int64)]
# Confirm the source row map remains exact for the specific old-cut witness.

# Save a minimal composition candidate; target arrays remain explicitly preflip diagnostic only.
OUT.mkdir(parents=True,exist_ok=False)
np.savez_compressed(OUT/'fat-D411-379-plus-comp1-run041-composed.npz',
 points_world_m=outpoints,points_world_m_f32=outp32,tetrahedra=outt,material_indices=outm,boundary_faces=actual,
 preflip_run041_desired_boundary_faces=mapped_target,
 input_node_to_output_node=fullmap,output_node_to_input_node=np.r_[np.asarray(tr['output_node_to_input_node'],np.int64),np.full(3,-1,np.int64)],
 retained_input_cell_rows=kept_base_rows,input_cell_to_output_cell=input_to_output,output_cell_to_input_cell=output_to_input,
 source_D411_transfer_path=np.asarray([str(TRANSFER)]),source_D411_transfer_sha256=np.asarray([TRANSFER_SHA]),
 source_comp1_run041_path=np.asarray([str(COMP1)]),source_comp1_run041_sha256=np.asarray([COMP1_SHA]),
 source_final_typed_D_path=np.asarray([str(FINAL_D)]),source_final_typed_D_sha256=np.asarray([FINAL_D_SHA]),
 source_comp1_replacement_start=np.asarray([replacement_start],np.int64),source_comp1_output_cell_rows=comp1_rows_out,
 source_comp1_added_point_rows=np.arange(N,N+3,dtype=np.int64),source_base406_cell_rows=output_to_input)
# Fresh readback and concise receipt.
cand=OUT/'fat-D411-379-plus-comp1-run041-composed.npz';z=np.load(cand,allow_pickle=False)
if not np.array_equal(z['tetrahedra'],outt) or not np.array_equal(z['points_world_m_f32'].view(np.uint32),outp32.view(np.uint32)):raise ValueError('readback differs')
report={
 'schema':'h706.d411-379-plus-run041-comp1-volume-composition.v1','status':'PASS_D411_QUOTIENT_COMPOSED_WITH_DISJOINT_COMP1_PATCH',
 'script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),'candidate_path':str(cand),'candidate_sha256':sha(cand),
 'base406_path':str(BASE),'base406_sha256':sha(BASE),'d411_transfer_path':str(TRANSFER),'d411_transfer_sha256':sha(TRANSFER),
 'd411_transfer_report_sha256':sha(TRANSFER_REPORT),'d411_independent_413_receipt_sha256':sha(TRANSFER_INDEPENDENT),
 'comp1_run041_path':str(COMP1),'comp1_run041_sha256':sha(COMP1),'comp1_run041_report_sha256':sha(COMP1_REPORT),
 'comp1_independent_419_sha256':sha(COMP1_INDEPENDENT),'comp1_shell_sha256':sha(COMP1_SHELL),
 'final_typed_D411_comp1_flips_path':str(FINAL_D),'final_typed_D411_comp1_flips_sha256':sha(FINAL_D),
 'final_D_actual_697_receipt_sha256':sha(FINAL_D_697),
 'input406_points':N,'output_points':int(len(outp32)),'appended_comp1_points':3,
 'input406_cells':int(len(baset)),'run041_retained_source_cells':int(len(ret)),
 'run041_comp1_replaced_base_cells':int(len(comp_rows)),'run041_new_tets':int(len(replacement)),
 'd411_operation_changed_source_cells':int(len(exact_base_rows)),'d411_operation_deleted_source_cells':int(np.count_nonzero(inmap<0)),
 'merged_output_cells':int(len(outt)),'retained_and_comp1_cell_count_identity':int(len(outt)),
 'full_boundary_equals_379_quotient_of_run041_actual_boundary':True,
 'preflip_run041_target_residual_after_quotient':{'missing_oriented_records':int(missing),'extra_oriented_records':int(extra)},
 'final_D411_comp1_three_flip_target_rebind':'PENDING; preflip target array retained under explicitly diagnostic field',
 'all_changed_D411_tets_exact_positive':True,'all_comp1_replacement_tets_exact_positive':True,
 'whole_output_nonpositive_by_float64_measurement':int(np.count_nonzero(newdet<=0)),
 'whole_output_native_quality_failures':int(np.count_nonzero(~newgood)),
 'retained_old_good_regressions':int(len(regress)),'comp1_replaced_old_good_cells':comp1_replaced_old_good,'d411_deleted_old_good_cells':d411_deleted_old_good,
 'comp0_old_cut_extra_cell_799862':{'base_cell_row':COMP0_EXTRA_CELL,'d411_transfer_output_cell_row':comp0_output_row,'composed_output_cell_row':comp0_merged_row,'cell_connectivity_changed_by_D411':bool(comp0_was_changed) if comp0_was_changed is not None else None,'tetrahedron_after_D411':comp0_in_transfer.tolist(),'present':bool(comp0_merged_row>=0)},
 'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN',
 'limitations':['This is a source-only volume composition of the validated D411 endpoint quotient and the disjoint run041 comp1 patch. The final D surface includes three additional 2-2 flips and requires a conforming volume-side update. The run041 stored target snapshot is explicitly preflip and produces the recorded residual; it is not the final typed D411/comp1/flip target. No mass mapping, native admission, or physical execution is claimed.'],
 'elapsed_s':time.monotonic()-start}
(OUT/'receipt.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'candidate':str(cand),'sha256':report['candidate_sha256'],'points':len(outp32),'tets':len(outt),'comp1_tets':len(replacement),'target_missing':missing,'target_extra':extra,'native_bad':int(np.count_nonzero(~newgood)),'retained_oldgood_regressions':len(regress),'comp1_replaced_oldgood':comp1_replaced_old_good,'d411_deleted_oldgood':d411_deleted_old_good,'comp0_799862':report['comp0_old_cut_extra_cell_799862'],'elapsed_s':round(report['elapsed_s'],2)},sort_keys=True))
