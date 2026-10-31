"""Apply frozen D309 through indexed209 owner directly to final I399, with exact-key join."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def zload(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
owner=module('indexed404',H/'transfer_surface_contractions_indexed_379_root.py');m=owner.m
reg=module('exact404',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
out=H/'combined-D309-I399-volume-404-root';out.mkdir(exist_ok=False);start=time.monotonic()
root=H/'d-shortedge-quotient-385-run006'
prepath=root/'preflight.json';pre=json.loads(prepath.read_text())
assert sha(prepath)=='ce18c8b27ab4fbea4f956f13910cb8d2bc7fe5be08f2a7ae85f970b296d8d0a5'
sidepath=root/'accepted-volume-star-map-001.npz';assert sha(sidepath)=='e4ebb7f243f1969312ab826082c7d1ebc68681bd6607ba5a1369cbde8cbe844e'
side=zload(sidepath)
ledger=root/'surface-run001/accepted-operations.jsonl';assert sha(ledger)=='f4ece3d374e877dd26d20ec8df9313af160e4c718a59bb2da446871ef5a4c958'
ops=[json.loads(x) for x in ledger.read_text().splitlines()];assert len(ops)==309
basepath=Path(pre['source_pins']['current_Fat_rebound_8915']['path']);assert sha(basepath)=='8915c3c7a58f8622d0568da5006bf4b7b0f2832f802b65e29bd144e95f11001d'
base=zload(basepath)
currentpath=H/'paired-I-interface-399-root/iteration002/batch/fat-coupled-I-quotient-volume-proposal.npz'
assert sha(currentpath)=='40ecc1079ead7b78e86fb5eac24c5cd6de80bb35e41004fae5130995fc1157b1'
cur=zload(currentpath);p=cur['points_world_m'];t=cur['tetrahedra'];mat=cur['material_indices']
assert np.array_equal(p.view(np.uint8),base['points_world_m'].view(np.uint8))
assert np.all(mat==16)
Dp=Path(pre['source_pins']['current_D']['path']);assert sha(Dp)==pre['source_pins']['current_D']['sha256']
dz=zload(Dp);sp=dz['points_world_m_f32'];sf=dz['faces']
seed=zload(root/'identity-seed-D.npz')
assert np.array_equal(seed.get('points_world_m_f32',seed.get('points_world_m')),sp) and np.array_equal(seed['faces'],sf)
raw=root/'surface-run001/candidate-D-short-edge-collapse-raw.npz';assert sha(raw)=='f8220c4a021acca0d15eea29c02ce7b51f7102954e242e8403ebf8adf9f971b7'
rz=zload(raw)
allold=side['D_operation_volume_star_current_cell_rows'];assert len(allold)==len(np.unique(allold))==1415
keys={tuple(map(int,q)):i for i,q in enumerate(t)};assert len(keys)==len(t)
joined=np.asarray([keys[tuple(map(int,q))] for q in base['tetrahedra'][allold]],np.int64)
assert np.array_equal(t[joined],base['tetrahedra'][allold]) and np.array_equal(mat[joined],base['material_indices'][allold])
del keys
bd=owner.boundary(t);look={owner.key(x):i for i,x in enumerate(p)};assert len(look)==len(p)
flat=t.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
v={'p':p,'t':t.copy(),'mat':mat,'base_boundary':bd,'boundary_keys':{tuple(sorted(map(int,f))) for f in bd},'look':look,'active':np.ones(len(t),bool),'map':np.arange(len(p)),'node_incidence_order':order,'node_incidence_offsets':offset,'node_incidence_added':{}}
sd=np.arange(len(sp));records=[];exactdelta=0
old_det,old_q=m.measurements(p,t);assert np.all(old_det>0)
before=int(np.sum(~m.passes(old_det,old_q)))
voff=side['D_operation_volume_star_offsets']
for oi,op in enumerate(ops):
 assert op['collapsed_input_point_row']==side['D_operation_collapsed_point_rows'][oi] and op['retained_input_point_row']==side['D_operation_retained_point_rows'][oi]
 trial,why=owner.try_op(v,sp,sf,op);assert trial is not None,(oi,why)
 a,b=map(int,voff[oi:oi+2]);expected=joined[a:b]
 assert np.array_equal(np.sort(trial['rows']),np.sort(expected))
 assert trial['retired_volume_node']==side['D_operation_volume_drop_point_rows'][oi] and trial['kept_volume_node']==side['D_operation_volume_keep_point_rows'][oi]
 oldsum=sum(reg.exact_star_determinants(p,trial['old']));newsum=sum(reg.exact_star_determinants(p,trial['new'][~trial['deleted']]))
 delta=int(newsum-oldsum);exactdelta+=delta
 v['t'][trial['rows']]=trial['new'];v['active'][trial['rows'][trial['deleted']]]=False
 v['map'][trial['retired_volume_node']]=trial['kept_volume_node']
 v['node_incidence_added'].setdefault(trial['kept_volume_node'],set()).update(map(int,trial['rows'][~trial['deleted']]))
 sd[op['collapsed_input_point_row']]=op['retained_input_point_row']
 records.append({'operation':oi,'source_8915_cell_rows':allold[a:b].tolist(),'source_I399_cell_rows':expected.tolist(),'kept_volume_node':trial['kept_volume_node'],'retired_volume_node':trial['retired_volume_node'],'removed_cells':trial['removed_cells'],'bad_before':trial['bad_before'],'bad_after':trial['bad_after'],'exact_F32_det6_delta':str(delta)})
wanted=sd[sf];wanted=wanted[~np.any(np.diff(np.sort(wanted,axis=1),axis=1)==0,axis=1)]
assert np.array_equal(owner.boundary_identity(wanted),owner.boundary_identity(rz['faces']))
tt=v['t'][v['active']];newbd=owner.boundary(tt)
expected=v['map'][bd];expected=expected[~np.any(np.diff(np.sort(expected,axis=1),axis=1)==0,axis=1)]
assert np.array_equal(owner.boundary_identity(newbd),owner.boundary_identity(expected))
det,q=m.measurements(p,tt);assert np.all(det>0);after=int(np.sum(~m.passes(det,q)))
assert after==before-sum(x['bad_before']-x['bad_after'] for x in records)
assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
unchanged=np.ones(len(t),bool);unchanged[joined]=False
assert np.array_equal(v['t'][unchanged],t[unchanged]) and np.all(v['active'][unchanged])
payload=dict(cur);payload['D309_previous_retained_input_cell_rows']=payload.get('retained_input_cell_rows',np.empty(0,np.int64))
payload.update(tetrahedra=tt,material_indices=mat[v['active']],boundary_faces=newbd,retained_input_cell_rows=np.flatnonzero(v['active']),source_D_vertex_quotient=sd,D309_source_8915_affected_cell_rows=allold,D309_source_I399_affected_cell_rows=joined,D309_input_node_quotient=v['map'])
outfile=out/'fat-D309-I399-full-domain-volume-proposal.npz';np.savez_compressed(outfile,**payload)
report={'status':'POSITIVE_VOLUME_QUOTIENT_PROPOSAL_REQUIRES_GEOMETRY_AND_MASS_QUALIFICATION','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','input':str(currentpath),'input_sha256':sha(currentpath),'source_8915':str(basepath),'source_8915_sha256':sha(basepath),'source_D':str(Dp),'source_D_sha256':sha(Dp),'D_operation_ledger':str(ledger),'D_operation_ledger_sha256':sha(ledger),'accepted_volume_star_map':str(sidepath),'accepted_volume_star_map_sha256':sha(sidepath),'candidate':str(outfile),'candidate_sha256':sha(outfile),'bad_before':before,'bad_after':after,'nonpositive':0,'cells_before':len(t),'cells_after':len(tt),'point_coordinates_bit_identical':True,'all_1415_old_stars_exact_key_joined':True,'all_unrelated_cells_and_materials_preserved':True,'boundary_equals_exact_D_quotient':True,'exact_F32_det6_volume_delta':str(exactdelta),'source_D_quotient_matches_frozen_raw_candidate':True,'operations':records,'producer_sha256':sha(__file__),'indexed_owner_sha256':sha(H/'transfer_surface_contractions_indexed_379_root.py'),'elapsed_s':time.monotonic()-start,'limitation':'New D/I joint context, complete typed current target, mass transfer and native quality admission remain mandatory.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='operations'}),flush=True)
