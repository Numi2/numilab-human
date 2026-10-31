#!/usr/bin/env python3
"""Independent readback of the D411 quotient + run041 comp1 exact-key composition."""
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'; BASE_SHA='5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211'
TRANSFER=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/fat-D411-379-current406-volume-proposal.npz'; TRANSFER_SHA='1d1b2e722f150bd9c840cb1090de53bf6c7ad72cd046a019a1f33fc99f5a5824'
SIDE=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/operation-cell-star-map.npz'; SIDE_SHA='153d6fb0178a964b16c493233ef51df0e43b9e1925ed99c7231eb7a4bb7372b0'
TRANSFER_INDEP=H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/independent-413-v2/independent-verification.json'; TRANSFER_INDEP_SHA='5ed728315b30e1155364218f2f1fe5886c6f37017658a822ff09e14d196a9be6'
COMP=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/connected-fixed-surface-volume-proposal.npz'; COMP_SHA='e11249bd297a7e08ae8d74fe406439289dfaf2f1f3069d1ebf83876e3d170a15'
COMP_REP=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/independent-full-audit-419.json'; COMP_REP_SHA='40314094d4eb66505236e4ec712f1528dbd63b83d0af40469f47134e1e5da6dd'
SHELL=H/'fat614-yy-volume-replay-004/run-037-D407-target-cavity-prep/connected-cavity-shells.npz'; SHELL_SHA='d0b813675dbbf5fb7a8ef41b0b961b67f104de769d09afab3ae3e5c7aeb7e089'
CAND=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/fat-D411-379-plus-comp1-run041-composed.npz'; CAND_SHA='14f06beaf20a2f78396585e7868df831cff4b9f27e3401e271d7643175073a77'
PROD=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/receipt.json'; PROD_SHA='1fd5a5719ca1e085269ee51467b5cb72ee224106002bb3fc866f1b5b5c52087d'
OWNER=H/'transfer_surface_contractions_indexed_379_root.py'; OWNER_SHA='ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37'
REG=H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py'
OUT=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/independent-415'
COMP0=799862
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def facectr(x):return Counter(tuple(map(int,r)) for r in np.asarray(x,np.int64))
for p,h in ((BASE,BASE_SHA),(TRANSFER,TRANSFER_SHA),(SIDE,SIDE_SHA),(TRANSFER_INDEP,TRANSFER_INDEP_SHA),(COMP,COMP_SHA),(COMP_REP,COMP_REP_SHA),(SHELL,SHELL_SHA),(CAND,CAND_SHA),(PROD,PROD_SHA),(OWNER,OWNER_SHA)):
 if sha(p)!=h:raise ValueError(f'input changed: {p}')
owner=mod('owner415',OWNER);m=owner.m;reg=mod('exact415',REG);tic=time.monotonic()
b=np.load(BASE,allow_pickle=False);t=np.load(TRANSFER,allow_pickle=False);s=np.load(SIDE,allow_pickle=False);c=np.load(COMP,allow_pickle=False);sh=np.load(SHELL,allow_pickle=False);o=np.load(CAND,allow_pickle=False);pr=json.loads(PROD.read_text())
p=np.asarray(b['points_world_m_f32'],np.float32);base_t=np.asarray(b['tetrahedra'],np.int64);base_m=np.asarray(b['material_indices'],np.int32)
N=len(p);comp_points=np.asarray(c['points_world_m'],np.float32);comp_t=np.asarray(c['tetrahedra'],np.int64);comp_m=np.asarray(c['material_indices'],np.int32);ret=np.asarray(c['retained_input_cell_rows'],np.int64)
assert len(comp_points)==N+3 and np.array_equal(comp_points[:N].view(np.uint32),p.view(np.uint32))
assert np.array_equal(comp_t[:len(ret)],base_t[ret]) and np.array_equal(comp_m[:len(ret)],base_m[ret])
cr=np.asarray(sh['input_cell_rows'],np.int64);assert len(cr)==2113 and np.array_equal(np.sort(np.setdiff1d(np.arange(len(base_t)),ret)),np.sort(cr))
node=np.arange(len(comp_points),dtype=np.int64);node[:N]=np.asarray(t['input_node_to_output_node'],np.int64)
mapped=node[comp_t];order=np.sort(mapped,axis=1);deg=np.any(order[:,1:]==order[:,:-1],axis=1);keep=~deg
exp_t=mapped[keep];exp_m=comp_m[keep];exp_pts=comp_points
assert np.array_equal(o['tetrahedra'],exp_t) and np.array_equal(o['material_indices'],exp_m)
assert np.array_equal(o['points_world_m_f32'].view(np.uint32),exp_pts.view(np.uint32))
assert len(exp_t)==819309 and len(exp_pts)==253223
trret=np.asarray(t['retained_input_cell_rows'],np.int64);inmap=np.full(len(base_t),-1,np.int64);inmap[trret]=np.arange(len(trret),dtype=np.int64)
mask=inmap[ret]>=0;expected_transfer_rows=inmap[ret[mask]]
assert not np.any(deg[:len(ret)][mask]) and np.all(deg[:len(ret)][~mask])
assert np.array_equal(exp_t[:int(mask.sum())],t['tetrahedra'][expected_transfer_rows])
repl=np.arange(len(ret),len(comp_t),dtype=np.int64);repl_out=int(mask.sum())
assert len(repl)==2212 and np.array_equal(exp_t[repl_out:],comp_t[repl]) and np.array_equal(exp_m[repl_out:],comp_m[repl])
assert np.array_equal(o['retained_input_cell_rows'],ret[mask])
# Independently recompute oriented actual boundary and compare with transformed run041 boundary.
actual=owner.boundary(exp_t);old=owner.boundary(comp_t);oldm=node[old];srt=np.sort(oldm,axis=1);valid=~np.any(srt[:,1:]==srt[:,:-1],axis=1);expected=oldm[valid]
assert facectr(owner.boundary_identity(actual))==facectr(owner.boundary_identity(expected))
assert np.array_equal(o['boundary_faces'],actual)
# Recompute the explicitly preflip target residual.
target=node[np.asarray(c['desired_boundary_faces'],np.int64)];srt=np.sort(target,axis=1);target=target[~np.any(srt[:,1:]==srt[:,:-1],axis=1)]
a=facectr(owner.boundary_identity(actual));q=facectr(owner.boundary_identity(target));missing=sum((q-a).values());extra=sum((a-q).values())
assert missing==117 and extra==181
# Exact-positive checks for all D411 changed survivors and all comp1 replacement tets.
side_changed=np.unique(np.asarray(s['operation_changed_cell_rows'],np.int64));in_to_out=np.asarray(o['input_cell_to_output_cell'],np.int64);kept_changed=side_changed[in_to_out[side_changed]>=0]
changed_det=reg.exact_star_determinants(exp_pts,exp_t[in_to_out[kept_changed]])
comp1_det=reg.exact_star_determinants(exp_pts,exp_t[repl_out:])
assert all(v>0 for v in changed_det) and all(v>0 for v in comp1_det)
# Independently recalculate float64 signs/quality and compare the output defects.
det,qval=m.measurements(exp_pts,exp_t);good=m.passes(det,qval)
assert np.all(det>0) and int(np.count_nonzero(~good))==24174
od,oq=m.measurements(p,base_t);og=m.passes(od,oq);retained=np.asarray(o['retained_input_cell_rows'],np.int64)
assert np.count_nonzero(og[retained] & ~good[:len(retained)])==0
# Verify the fourth comp0 old-cut source row and its exact output row mapping.
assert int(inmap[COMP0])==793051 and int(o['input_cell_to_output_cell'][COMP0])==790948
assert np.array_equal(exp_t[790948],base_t[COMP0])
OUT.mkdir(parents=True,exist_ok=False)
report={'schema':'h706.d411-plus-comp1-independent-composition-replay.v1','status':'PASS_INDEPENDENT_D411_COMP1_EXACT_KEY_COMPOSITION',
 'verifier_path':str(Path(__file__)),'verifier_sha256':sha(Path(__file__)),'candidate_path':str(CAND),'candidate_sha256':sha(CAND),'producer_receipt_sha256':sha(PROD),
 'base406_sha256':sha(BASE),'d411_transfer_sha256':sha(TRANSFER),'d411_413_receipt_sha256':sha(TRANSFER_INDEP),'comp1_run041_sha256':sha(COMP),'comp1_419_receipt_sha256':sha(COMP_REP),'comp1_shell_sha256':sha(SHELL),
 'input_point_count':N,'output_point_count':len(exp_pts),'comp1_appended_points':3,'input_cell_count':len(base_t),'output_cell_count':len(exp_t),'d411_deleted_cells':int(np.count_nonzero(inmap<0)),'comp1_replacement_tets':len(repl),
 'retained_cell_order_matches_exact_379_transfer':True,'comp1_replacement_tets_unchanged':True,'actual_boundary_equals_exact_quotient_of_run041':True,
 'preflip_target_residual_missing':int(missing),'preflip_target_residual_extra':int(extra),'changed_D411_exact_positive':True,'comp1_replacement_exact_positive':True,
 'whole_volume_float64_nonpositive':int(np.count_nonzero(det<=0)),'whole_volume_native_quality_failures':int(np.count_nonzero(~good)),'retained_old_good_regressions':0,
 'comp0_extra_source_cell_799862':{'d411_transfer_output_row':int(inmap[COMP0]),'composed_output_row':int(in_to_out[COMP0]),'connectivity_unchanged':True,'tet':exp_t[int(in_to_out[COMP0])].tolist()},
 'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','elapsed_s':time.monotonic()-tic,
 'limitations':['This verifies D411 quotient plus the disjoint run041 comp1 volume patch. The final three D 2-2 flips remain a separate volume-side conformity change; target residual is measured against the preflip run041 snapshot and is not final.']}
path=OUT/'independent-verification.json';path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'receipt':str(path),'sha256':sha(path),'tets':len(exp_t),'native_bad':report['whole_volume_native_quality_failures'],'missing':missing,'extra':extra,'elapsed_s':round(report['elapsed_s'],2)},sort_keys=True))
