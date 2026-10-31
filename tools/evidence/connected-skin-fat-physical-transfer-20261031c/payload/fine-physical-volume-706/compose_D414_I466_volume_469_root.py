"""Compose disjoint authenticated D and paired-I volume quotients by exact parent rows."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=mod('compose468',H/'transfer_surface_contractions_indexed_379_root.py');R=mod('exact468',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
out=H/'combined-D414-I466-volume-469-root';out.mkdir(exist_ok=False);start=time.monotonic()
dp=H/'d-shortedge-quotient-414-batch001-volume-transfer/run002/fat-D414-surface-quotient-volume-proposal.npz'
assert sha(dp)=='0cfd2d78372597e5ff0bad547b10321d2b3e953ee7a20b899017bb197d146c0a'
ip=H/'paired-I-quality-exchange-466-root/exchange462/report.json';ir=json.loads(ip.read_text());r=next(x for x in ir['roles'] if x['role']=='Fat');bp=Path(r['input']);assert sha(bp)==r['input_sha256']
with np.load(dp) as z:data={k:z[k] for k in z.files}
with np.load(bp) as z:base={k:z[k] for k in z.files}
p=data['points_world_m'];dt=data['tetrahedra'];bt=base['tetrahedra'];assert np.array_equal(p.view(np.uint32),base['points_world_m'].view(np.uint32))
assert np.all(data['material_indices']==16) and np.all(base['material_indices']==16)
mapping=data['input_cell_to_output_cell'];assert mapping.shape==(len(bt),)
mp=np.arange(len(p));ops=[];oldsum=newsum=0;allaffected=set()
for entry,op in zip(r['operations'],ir['operations']):
 rows=np.asarray(entry['source_cell_rows'],np.int64);mapped=mapping[rows];assert np.all(mapped>=0)
 assert np.array_equal(dt[mapped],bt[rows]),'D and I changed cells are not disjoint'
 assert np.all(base['material_indices'][rows]==data['material_indices'][mapped])
 keep=entry['kept_volume_node'];drop=entry['retired_volume_node'];assert mp[keep]==keep and mp[drop]==drop
 assert np.array_equal(rows,np.flatnonzero(np.any(bt==drop,axis=1)))
 old=dt[mapped];new=np.where(old==drop,keep,old);degen=np.any(np.diff(np.sort(new,axis=1),axis=1)==0,axis=1)
 d0,q0=T.m.measurements(p,old);d1,q1=T.m.measurements(p,new[~degen]);g0=T.m.passes(d0,q0);g1=T.m.passes(d1,q1)
 assert np.all(d1>0) and int((~g0).sum())==entry['bad_before'] and int((~g1).sum())==entry['bad_after']
 assert int(np.count_nonzero(g0[~degen]&~g1))==entry['new_retained_quality_failures']
 exact0=R.exact_star_determinants(p,old);exact1=R.exact_star_determinants(p,new[~degen]);assert all(x>0 for x in exact1)
 oldsum+=sum(exact0);newsum+=sum(exact1);mp[drop]=keep;allaffected.update(map(int,mapped))
 ops.append({'operation':entry['operation'],'base_parent_rows':rows.tolist(),'D_current_parent_rows':mapped.tolist(),'kept_volume_node':keep,'retired_volume_node':drop,'old_bad':entry['bad_before'],'new_bad':entry['bad_after'],'new_retained_quality_failures':entry['new_retained_quality_failures']})
nt=mp[dt];degen=np.any(np.diff(np.sort(nt,axis=1),axis=1)==0,axis=1);keep=~degen;nt=nt[keep]
bound=T.boundary(nt);expected=mp[T.boundary(dt)];dd=np.any(np.diff(np.sort(expected,axis=1),axis=1)==0,axis=1)
assert np.array_equal(T.boundary_identity(bound),T.boundary_identity(expected[~dd]))
assert len(np.unique(np.sort(nt,axis=1),axis=0))==len(nt)
det,q=T.m.measurements(p,nt);det0,q0=T.m.measurements(p,dt);bad0=int(np.count_nonzero(~T.m.passes(det0,q0)));bad1=int(np.count_nonzero(~T.m.passes(det,q)))
assert np.all(det>0) and bad1==bad0-sum(o['old_bad']-o['new_bad'] for o in ops)==20443
# Source I paired independently with Skin; their integer-lattice volume deltas cancel.
typed=H/'paired-I-quality-exchange-466-root/typed463001/receipt.json';tr=json.loads(typed.read_text())
fatrole=next(x for x in tr['roles'] if x['role']=='Fat');skinrole=next(x for x in tr['roles'] if x['role']=='Skin')
assert newsum-oldsum==int(fatrole['exact_signed_integer_volume_delta'])==-int(skinrole['exact_signed_integer_volume_delta'])
target=mp[data['desired_boundary_faces']];dd=np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1);target=target[~dd]
inputmap=np.full(len(dt),-1,np.int64);inputmap[keep]=np.arange(len(nt));retained=np.flatnonzero(keep)
data.update(tetrahedra=nt,material_indices=data['material_indices'][keep],boundary_faces=bound,desired_boundary_faces=target,
 retained_input_cell_rows=retained,input_cell_to_output_cell=inputmap,output_cell_to_input_cell=retained,
 current_geometry_parent_path=np.asarray([str(dp)]),current_geometry_parent_sha256=np.asarray([sha(dp)]),
 source_I_vertex_quotient=np.load(Path(r['output']))['source_I_vertex_quotient'],
 typed_I_candidate_path=np.asarray([tr['candidate']]),typed_I_candidate_sha256=np.asarray([tr['candidate_sha256']]),
 paired_I_construction_report_path=np.asarray([str(ip)]),paired_I_construction_report_sha256=np.asarray([sha(ip)]))
candidate=out/'fat-D414-I466-composed-volume-proposal.npz';np.savez_compressed(candidate,**data)
np.savez_compressed(out/'operation-cell-map.npz',input_cell_to_output_cell=inputmap,output_cell_to_input_cell=retained,source_volume_node_quotient=mp,affected_D_current_cell_rows=np.asarray(sorted(allaffected),np.int64))
report={'status':'PASS_EXACT_DISJOINT_D414_I466_VOLUME_COMPOSITION','physical_admission':False,'native_steps':0,'D_parent':str(dp),'D_parent_sha256':sha(dp),'I_report':str(ip),'I_report_sha256':sha(ip),'candidate':str(candidate),'candidate_sha256':sha(candidate),'I_operations':len(ops),'operations':ops,'point_coordinates_bit_identical':True,'material_preserved':True,'exact_quotient_boundary':True,'nonpositive':0,'input_cells':len(dt),'output_cells':len(nt),'bad_before_D_current':bad0,'bad_after':bad1,'combined_Skin_Fat_integer_volume_delta':0,'construction_new_retained_quality_failures':sum(x['new_retained_quality_failures'] for x in ops),'current_D_typed_target_status':'REQUIRES_FINAL_D_LOADER_CONTEXT_AND_397_REBIND','source330_mass':'IMMUTABLE_COMPLETE_CURRENT_CONSERVATIVE_REBIND_REQUIRED','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='operations'}),flush=True)
