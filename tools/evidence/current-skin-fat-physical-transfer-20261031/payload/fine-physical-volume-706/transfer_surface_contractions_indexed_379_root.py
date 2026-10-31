"""Existing209 exact quotient predicates with optional cached node incidence.
No geometry, quality, ancestry or physical acceptance policy changes.
"""
"""Try authenticated surface endpoint contractions, including valid boundary-ear removal.
Only exact-compatible boundary patches are considered. No mass/native admission.
"""
from pathlib import Path
import numpy as np,json,hashlib,time,importlib.util,argparse
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cavity198',H/'fixed_boundary_volume_flip_198_root.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def key(p):return np.asarray(p,np.float32).view(np.uint32).tobytes()
def boundary(t):
 f=t[:,m.PAT].reshape(-1,3);sf=np.sort(f,axis=1);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));ss=sf[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 if np.any(ct>2):raise ValueError('nonmanifold tet face')
 inv=((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8))
 signs=1-2*(inv%2);sums=np.add.reduceat(signs[order],starts)
 if np.any(sums[ct==2]!=0):raise ValueError('inconsistent internal winding')
 rows=order[starts[ct==1]];return f[rows]
def boundary_identity(b):
 x=np.sort(b,axis=1);order=np.lexsort((x[:,2],x[:,1],x[:,0]));inv=((b[:,0]>b[:,1]).astype(np.int8)+(b[:,0]>b[:,2]).astype(np.int8)+(b[:,1]>b[:,2]).astype(np.int8))
 return np.c_[x[order],(1-2*(inv%2))[order]]
def read_volume(role):
 path=H/('fixed-boundary-volume-flips-203-root/'+role.lower()+'-batch001/pass-007/fixed-boundary-volume-proposal.npz')
 with np.load(path) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
 b=boundary(t);look={key(x):i for i,x in enumerate(p)}
 if len(look)!=len(p):raise ValueError('ambiguous coordinate crosswalk')
 return {'path':path,'p':p,'t':t.copy(),'mat':mat,'base_boundary':b,'boundary_keys':{tuple(sorted(map(int,f))) for f in b},'look':look,'active':np.ones(len(t),bool),'map':np.arange(len(p)),'accepted':[]}
def try_op(v,sp,sf,op):
 keep=int(op['retained_input_point_row']);drop=int(op['collapsed_input_point_row'])
 if key(sp[keep]) not in v['look'] or key(sp[drop]) not in v['look']:return None,'point_not_in_volume'
 u=v['look'][key(sp[keep])];w=v['look'][key(sp[drop])]
 if v['map'][w]!=w or v['map'][u]!=u:return None,'dependent_contraction'
 # Do not silently apply DP004 faces to a volume with a different boundary.
 for fr in op['retired_parent_face_rows']:
  try:tri=[v['look'][key(x)] for x in sp[sf[int(fr)]]]
  except KeyError:return None,'source_face_point_not_in_volume'
  if tuple(sorted(tri)) not in v['boundary_keys']:return None,'requires_DP_local_cavity_retriangulation'
 if 'node_incidence_order' in v:
  order=v['node_incidence_order'];offset=v['node_incidence_offsets']
  base=order[offset[w]:offset[w+1]]//4
  added=np.asarray(sorted(v['node_incidence_added'].get(w,())),np.int64)
  rows=np.unique(np.r_[base,added])
  rows=rows[v['active'][rows]&np.any(v['t'][rows]==w,axis=1)]
 else:
  rows=np.flatnonzero(v['active']&np.any(v['t']==w,axis=1))
 if not len(rows):return None,'unused_source_vertex'
 old=v['t'][rows].copy();new=np.where(old==w,u,old)
 sortedn=np.sort(new,axis=1);deleted=np.any(sortedn[:,1:]==sortedn[:,:-1],axis=1)
 remains=new[~deleted]
 if not len(remains) and len(rows)==int(np.count_nonzero(v['active'])):return None,'empty_whole_volume'
 od,oq=m.measurements(v['p'],old);nd,nq=m.measurements(v['p'],remains)
 if np.any(nd<=0):return None,'F32_volume_inversion'
 # Certify every new sign on the exact Float32 integer lattice.
 used=np.unique(remains);lk={int(x):i for i,x in enumerate(used)};ip=m.integer_points(v['p'][used])
 if any(m.exact_det(ip,[lk[int(x)] for x in tet])<=0 for tet in remains):return None,'exact_F32_nonpositive'
 oldgood=m.passes(od,oq);newgood=m.passes(nd,nq)
 if np.any(oldgood[~deleted]&~newgood):return None,'new_quality_failure'
 if sum(~newgood)>sum(~oldgood):return None,'quality_count_increase'
 return {'rows':rows,'old':old,'new':new,'deleted':deleted,'kept_volume_node':u,'retired_volume_node':w,'removed_cells':int(sum(deleted)),'bad_before':int(sum(~oldgood)),'bad_after':int(sum(~newgood))},'PASS'
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 dp=H/'fixed-boundary-quality-topology-196/dp-batch-004';ledger=H/'fixed-boundary-quality-topology-196/collapse-batch-002/accepted-operations.jsonl'
 ops=[json.loads(l) for l in ledger.read_text().splitlines()];surfaces={}
 for role in ('O','I','D'):
  with np.load(dp/('candidate-'+role+'-fixed-boundary-dp-raw.npz')) as z:surfaces[role]=(z['points_world_m'].astype(np.float32),z['faces'])
 volumes={role:read_volume(role) for role in ('Skin','Fat')};accepted=[];rejected=[]
 for oi,op in enumerate(ops):
  role=op['role'];targets=('Skin',) if role=='O' else ('Fat',) if role=='D' else ('Skin','Fat')
  trials={};failure=None
  for vr in targets:
   result,why=try_op(volumes[vr],*surfaces[role],op)
   if result is None:failure=vr+':'+why;break
   trials[vr]=result
  if failure is not None:rejected.append({'operation':oi,'role':role,'reason':failure});continue
  for vr,r in trials.items():
   v=volumes[vr];v['t'][r['rows']]=r['new'];v['active'][r['rows'][r['deleted']]]=False;v['map'][r['retired_volume_node']]=r['kept_volume_node']
   v['accepted'].append({'operation':oi,**{k:r[k] for k in ('kept_volume_node','retired_volume_node','removed_cells','bad_before','bad_after')}})
  accepted.append({'operation':oi,'role':role,'surface_patch_id':op['patch_id']})
 roles=[]
 for role,v in volumes.items():
  t=v['t'][v['active']];b=boundary(t)
  expected=v['map'][v['base_boundary']];se=np.sort(expected,axis=1);expected=expected[~np.any(se[:,1:]==se[:,:-1],axis=1)]
  if not np.array_equal(boundary_identity(expected),boundary_identity(b)):raise ValueError(role+' quotient boundary does not match exact surface contraction')
  used=np.unique(t);newindex=np.full(len(v['p']),-1,np.int64);newindex[used]=np.arange(len(used));ot=newindex[t];p=v['p'][used]
  det,q=m.measurements(p,ot);assert np.all(det>0)
  out=a.output/(role.lower()+'-surface-contraction-volume-proposal.npz')
  np.savez_compressed(out,points_world_m=p,tetrahedra=ot,material_indices=v['mat'][v['active']],boundary_faces=newindex[b],retained_input_cell_rows=np.flatnonzero(v['active']),input_node_to_output_node=newindex,output_node_to_input_node=used,source_surface_contraction_rows=np.asarray([r['operation'] for r in v['accepted']],np.int64))
  roles.append({'role':role,'input_volume':str(v['path']),'input_sha256':sha(v['path']),'output':str(out),'output_sha256':sha(out),'boundary_matches_contraction_exact':True,'positive_F32_cells':len(ot),'nodes':len(p),'cells':len(ot),'contracted_nodes':len(v['accepted']),'removed_cells':len(v['t'])-len(ot),'normalized_det_failures':int(sum(q<1e-5)),'absolute_volume_failures':int(sum(det/6<=1e-18)),'union_failures':int(sum(~m.passes(det,q))),'minimum_normalized_det':float(q.min()),'minimum_volume_m3':float(det.min()/6),'volume_operations':v['accepted']})
 report={'status':'POSITIVE_VOLUME_QUOTIENT_PROPOSAL_REQUIRES_GEOMETRY_AND_MASS_QUALIFICATION','physical_admission':False,'native_steps':0,'ledger':str(ledger),'ledger_sha256':sha(ledger),'accepted_surface_operations':accepted,'rejected_operations':rejected,'roles':roles,'script_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-start,'limitations':['This is a subset of collapse002 compatible with the pre-DP physical boundary, not the full DP004/collapse002 surface.','Subset geometry intersections require exact audit before use.','Mass mapping, source ancestry regeneration and native geometry admission remain incomplete.']}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({**report,'accepted_surface_operations':len(accepted),'rejected_operations':len(rejected),'roles':[{k:v for k,v in r.items() if k!='volume_operations'} for r in roles]},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);run(ap.parse_args())
