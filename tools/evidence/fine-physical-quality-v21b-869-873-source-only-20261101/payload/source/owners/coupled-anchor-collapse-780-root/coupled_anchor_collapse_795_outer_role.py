#!/usr/bin/env python3
"""792: single-role optimized-anchor O/D edge collapse and independent replay.

O is paired only with Skin; D is paired only with Fat. Current back rows,
rims and registered routes are immutable. Single-role exact volume change is
bounded and reported; it is not falsely constrained to cancel.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
from collections import Counter
from itertools import product, combinations
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=HERE/'coupled_anchor_collapse_791_current_parent_root.py'
BASE_SHA='5d8b00a3ca57bbcb0c53ad1b92c78c97cb3ade5da441d3b5d71a7c3dfdb6fd76'
ENGINE_SHA='739e41b0fb4a3bd7d7eff2b9688657bad325d294185f7ae22a25af9aa2b186eb'
NATIVE_Q=1e-5
MIN_CELL_VOLUME_M3=1e-18
DET_TO_M3=2.0**(-447)/6.0
MOVE_CAP=.0005
EDGE_CAP=.005

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def import_pinned(path,digest,name):
 if sha(path)!=digest: raise RuntimeError('pinned source changed: '+str(path))
 s=importlib.util.spec_from_file_location(name,path)
 if not s or not s.loader: raise RuntimeError('cannot import '+str(path))
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

V=import_pinned(BASE,BASE_SHA,'coupled_anchor_791_for_792')
E=V.M
if sha(E.__file__)!=ENGINE_SHA: raise RuntimeError('781 engine hash mismatch')
S=V.G.G
T,A,B,Q,J=E.T,E.A,E.B,E.Q,E.J
SOURCE807=HERE.parent/'verify_current_source_evidence_807_root.py'
SOURCE807_SHA='9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62'
SOURCEPROOF=import_pinned(SOURCE807,SOURCE807_SHA,'source_evidence_807_for_795')
def source_evidence(state_path,state=None):
 proof=SOURCEPROOF.verify_current_source_evidence(state_path,state)
 if not str(proof.get('status','')).startswith('PASS_'):raise ValueError('807 current source evidence did not pass')
 if proof.get('physical_admission') is not False or int(proof.get('native_steps',-1))!=0:
  raise ValueError('807 source evidence unexpectedly carries admission')
 pins=proof.get('source_surfaces',{})
 if set(pins)!={'O','I','D'}:raise ValueError('807 proof lacks complete O/I/D sources')
 return proof

def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def pinned(ref,label):
 if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256'):
  raise ValueError(label+' missing path/SHA pin')
 p=Path(ref['path']).resolve()
 if not p.is_file() or sha(p)!=ref['sha256']:
  raise ValueError(label+' SHA mismatch: '+str(p))
 return p
def fkey(x): return np.ascontiguousarray(np.asarray(x,np.float32).reshape(3)).view(np.uint32).tobytes()
def ckey(t): return tuple(sorted(map(int,t)))
def boundary(t): return T.boundary(np.asarray(t,np.int64))
def exact_sum(p,t):
 t=np.asarray(t,np.int64)
 if not t.size:return 0
 nodes=np.unique(t.ravel());lut={int(v):i for i,v in enumerate(nodes)}
 ip=E.M.integer_points(np.asarray(p,np.float32)[nodes])
 return sum(E.M.exact_det(ip,[lut[int(v)] for v in row]) for row in t)
def exact_pos(p,t):
 t=np.asarray(t,np.int64)
 if not t.size:return True
 nodes=np.unique(t.ravel());lut={int(v):i for i,v in enumerate(nodes)}
 ip=E.M.integer_points(np.asarray(p,np.float32)[nodes])
 return all(E.M.exact_det(ip,[lut[int(v)] for v in row])>0 for row in t)
def anchor_order(tets,u):
 out=np.asarray(tets,np.int64).copy()
 for i,t in enumerate(out):
  pos=np.flatnonzero(t==u)
  if len(pos)!=1:raise ValueError('replacement misses retained anchor')
  j=int(pos[0])
  if j:
   out[i]=np.r_[t[j:],t[:j]]
   if j&1:out[i,2],out[i,3]=out[i,3],out[i,2]
 return out
def source_link(f,k,d):
 f=np.asarray(f,np.int64);kr=np.flatnonzero(np.any(f==k,axis=1));dr=np.flatnonzero(np.any(f==d,axis=1));er=np.intersect1d(kr,dr)
 if len(er)!=2:return False,{'reason':'edge_incidence','rows':er.tolist()}
 def lk(rows,want):
  out=set()
  for row in f[rows]:
   rest=sorted(set(map(int,row))-set(want))
   for i in range(len(rest)+1):out.update(combinations(rest,i))
  return out
 return (lk(kr,(k,))&lk(dr,(d,)))==lk(er,(k,d)),{'edge_face_rows':er.tolist()}
def volume_link(t,k,d):
 t=np.asarray(t,np.int64);kr=np.flatnonzero(np.any(t==k,axis=1));dr=np.flatnonzero(np.any(t==d,axis=1));er=np.intersect1d(kr,dr)
 if not len(er):return False
 def lk(rows,want):
  out=set()
  for row in t[rows]:
   rest=sorted(set(map(int,row))-set(want))
   for i in range(len(rest)+1):out.update(combinations(rest,i))
  return out
 return (lk(kr,(k,))&lk(dr,(d,)))==lk(er,(k,d))
def neighbors(x,r):
 axes=[]
 for v in np.asarray(x,np.float32).reshape(3):
  vals=[v];lo=hi=v
  for _ in range(int(r)):
   lo=np.nextafter(lo,np.float32(-np.inf),dtype=np.float32);hi=np.nextafter(hi,np.float32(np.inf),dtype=np.float32);vals.extend((lo,hi))
  axes.append(vals)
 return [np.asarray(z,np.float32) for z in product(*axes)]

def fit_single(role,x0,triangles,old_det,old_exact,local_det,regional_det,*,cap,ulp_radius=2):
 x0=np.asarray(x0,np.float32).reshape(3);tri0=np.asarray(triangles,np.float32).reshape(-1,3,3);cap=float(cap)
 if not np.isfinite(x0).all() or cap<=0 or not len(tri0) or not np.isfinite(tri0).all():return {'status':'REJECT_BAD_INPUT'}
 lengths=np.concatenate([np.linalg.norm(tri0[:,i].astype(np.float64)-tri0[:,j].astype(np.float64),axis=1) for i,j in ((0,1),(1,2),(2,0))])
 scale=max(cap,float(np.max(lengths)),float(np.max(np.linalg.norm(tri0.astype(np.float64)-x0.astype(np.float64),axis=2))),1e-12)
 tri=(tri0.astype(np.float64)-x0.astype(np.float64))/scale;a=tri[:,0]
 n=np.cross(tri[:,2]-tri[:,0],tri[:,1]-tri[:,0]);area=np.linalg.norm(n,axis=1)
 if np.any(area<=64*np.finfo(float).eps):return {'status':'REJECT_DEGENERATE_FACE'}
 fl=np.max(np.stack([np.linalg.norm(tri[:,0]-tri[:,1],axis=1),np.linalg.norm(tri[:,1]-tri[:,2],axis=1),np.linalg.norm(tri[:,2]-tri[:,0],axis=1)],axis=1),axis=1)**3
 vf=6*MIN_CELL_VOLUME_M3/scale**3;qm=NATIVE_Q*1.01;zero=np.zeros(3)
 h0=Q._normalized_margin(zero,n,a,fl,tri,vf,qm,area)
 if not np.isfinite(h0):return {'status':'REJECT_INITIAL_SLACK'}
 budget=min(.01*(float(local_det)/scale**3),1e-4*(float(regional_det)/scale**3))
 if budget<=0:return {'status':'REJECT_VOLUME_BUDGET'}
 old=float(old_det)/scale**3
 def delta(x):return float(np.sum(np.einsum('ij,ij->i',n,x[None,:]-a))-old)
 grad=np.sum(n,axis=0)
 cons=[
 {'type':'ineq','fun':lambda z:Q._quality_constraints(z[:3],z[3],n,a,fl,tri,vf,qm,area),'jac':lambda z:Q._quality_jacobian(z[:3],z[3],n,a,fl,tri,vf,qm,area)},
 {'type':'ineq','fun':lambda z:budget-delta(z[:3]),'jac':lambda z:np.r_[-grad,0.]},
 {'type':'ineq','fun':lambda z:budget+delta(z[:3]),'jac':lambda z:np.r_[grad,0.]},
 {'type':'ineq','fun':lambda z:(cap/scale)**2-float(np.dot(z[:3],z[:3])),'jac':lambda z:np.r_[-2*z[:3],0.]}
 ]
 res=minimize(lambda z:-float(z[3]),np.r_[zero,h0],jac=lambda z:np.array([0.,0.,0.,-1.]),method='SLSQP',
  bounds=[(-cap/scale,cap/scale)]*3+[(None,None)],constraints=cons,options={'maxiter':350,'ftol':1e-12,'disp':False})
 if res.x is None or not np.isfinite(res.x).all():return {'status':'REJECT_SOLVER','message':str(res.message)}
 x64=x0.astype(np.float64)+scale*res.x[:3];wins=[]
 for x in neighbors(np.asarray(x64,np.float32),ulp_radius):
  disp=float(np.linalg.norm(x.astype(np.float64)-x0.astype(np.float64)))
  if disp>cap:continue
  p=np.vstack((x,tri0.reshape(-1,3))).astype(np.float32);ntri=len(tri0)
  t=np.column_stack((np.zeros(ntri,np.int64),np.arange(1,3*ntri+1,dtype=np.int64).reshape(-1,3)))
  det,q=E.measurements(p,t)
  if np.any(det<=0) or not np.all(E.M.passes(det,q)) or not exact_pos(p,t):continue
  de=exact_sum(p,t)-int(old_exact);dm=de*DET_TO_M3
  if abs(dm)>budget*scale**3/6:continue
  wins.append((float(np.min(q)),-disp,x.copy(),int(de),float(dm),disp))
 if not wins:return {'status':'REJECT_NO_F32_CANDIDATE','message':str(res.message),'continuous_anchor_world_m':(x0.astype(np.float64)+scale*res.x[:3]).tolist()}
 best=max(wins,key=lambda z:(z[0],z[1]))
 return {'status':'PASS','role':role,'anchor_point_f32':best[2],'anchor_displacement_m':best[5],
  'minimum_new_native_q':best[0],'solver_success':bool(res.success),'solver_message':str(res.message),
  'exact_det_delta_integer':str(best[3]),'delta_volume_m3':best[4],
  'construction_q_margin_multiplier':1.01,'move_cap_m':cap,'native_q_gate_unchanged':NATIVE_Q,
  'minimum_cell_volume_m3_unchanged':MIN_CELL_VOLUME_M3,'volume_policy':'single-role signed delta, no paired-zero equation'}

def source_q(p,f):
 if not len(f):return True,1.
 q=np.asarray(J.score(np.asarray(p,np.float32),np.asarray(f,np.int64)),np.float64)
 return bool(np.isfinite(q).all() and np.all(q>=J.GATE)),float(np.min(q))
def normals(p,f):
 x=np.asarray(p,np.float32)[np.asarray(f,np.int64)].astype(np.float64)
 return np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
def prep_operation(surface,cache,role,k,d,protected,routes,backfat,cap):
 tissue='Skin' if role=='O' else 'Fat';sp=np.asarray(surface['points_world_m_f32'],np.float32);sf=np.asarray(surface['faces'],np.int64);k=int(k);d=int(d)
 ok,why=source_link(sf,k,d)
 if not ok:return {'status':'REJECT_SURFACE_LINK','witness':why}
 fr=np.union1d(np.flatnonzero(np.any(sf==k,axis=1)),np.flatnonzero(np.any(sf==d,axis=1)))
 verts=np.unique(sf[fr].ravel())
 if set(map(int,verts))&set(map(int,protected)):return {'status':'REJECT_BACK_RIM_ROUTE'}
 if {k,d}&set(map(int,routes)):return {'status':'REJECT_ROUTE'}
 v=cache;u=v['lookup'].get(fkey(sp[k]));w=v['lookup'].get(fkey(sp[d]))
 if u is None or w is None:return {'status':'REJECT_CROSSWALK'}
 kr=np.flatnonzero(np.any(v['t']==u,axis=1));dr=np.flatnonzero(np.any(v['t']==w,axis=1))
 if not volume_link(v['t'],u,w):return {'status':'REJECT_VOLUME_LINK'}
 rows=np.union1d(kr,dr);old=v['t'][rows]
 if set(np.unique(old).tolist())&set(map(int,backfat)):return {'status':'REJECT_ACTUAL_FAT_BACK_STAR'}
 mapped=np.where(old==w,u,old);dead=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1);new=mapped[~dead];parents=rows[~dead]
 if not len(new) or not np.all(np.any(new==u,axis=1)):return {'status':'REJECT_EMPTY_STAR'}
 if len({ckey(t) for t in new})!=len(new):return {'status':'REJECT_DUPLICATE_LOCAL_TET'}
 ordered=anchor_order(new,u);old_det=v['det'][rows];oldsum=float(old_det.sum());oldexact=exact_sum(v['p'],old)
 triangles=v['p'][ordered[:,1:]]
 fit=fit_single(tissue,sp[k],triangles,oldsum,oldexact,oldsum,float(v['det'].sum()),cap=cap)
 if fit.get('status')!='PASS':return {'status':'REJECT_ANCHOR_FIT','fit':fit}
 x=np.asarray(fit['anchor_point_f32'],np.float32);pnew=v['p'].copy();pnew[u]=x
 nd,nq=E.measurements(pnew,new)
 if np.any(nd<=0) or not np.all(E.M.passes(nd,nq)) or not exact_pos(pnew,new):return {'status':'REJECT_NATIVE_OR_POSITIVE'}
 delta=exact_sum(pnew,new)-oldexact;dm=delta*DET_TO_M3;budget=min(.01*oldsum,1e-4*float(v['det'].sum()))/6.
 if abs(dm)>budget:return {'status':'REJECT_VOLUME_BUDGET','delta_m3':dm,'budget_m3':budget}
 oldbad=int(np.count_nonzero(v['bad'][rows]));newbad=int(np.count_nonzero(~E.M.passes(nd,nq)))
 if oldbad-newbad<=0:return {'status':'REJECT_NO_STRICT_NATIVE_GAIN'}
 qmap=np.arange(len(sp),dtype=np.int64);qmap[d]=k;spout=sp.copy();spout[k]=x;fm=qmap[sf];valid=~np.any(np.diff(np.sort(fm,axis=1),axis=1)==0,axis=1);fparents=np.flatnonzero(valid);fout=fm[valid]
 if len({tuple(sorted(map(int,f))) for f in fout})!=len(fout):return {'status':'REJECT_DUPLICATE_SURFACE_FACE'}
 mappedstar=qmap[sf[fr]];usable=~np.any(np.diff(np.sort(mappedstar,axis=1),axis=1)==0,axis=1)
 oldn=normals(sp,sf[fr])[usable];newn=normals(spout,mappedstar[usable]);den=np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1)
 if np.any(den<=0) or np.any(np.einsum('ij,ij->i',oldn,newn)<.9*den):return {'status':'REJECT_SOURCE_NORMAL'}
 qok,minq=source_q(spout,mappedstar[usable])
 if not qok:return {'status':'REJECT_SOURCE_APEX','minimum':minq}
 return {'status':'PASS','keep':k,'drop':d,'role':role,'volume_role':tissue,'anchor_f32':x,
  'fit':fit,'surface_points':spout,'surface_faces':fout,'surface_parent_face_rows':fparents,
  'surface_full_endpoint_face_rows':fr,'surface_retired_face_rows':np.setdiff1d(fr,fparents),
  'surface_star_vertices':verts,'volume_endpoints':[int(u),int(w)],'volume_star_cells':rows,
  'volume_star_points':np.unique(old.ravel()),'replacement_tets':new,'replacement_parent_rows':parents,
  'replacement_materials':v['mat'][parents],'old_rows':rows,'old_bad':oldbad,'new_bad':newbad,
  'exact_delta_integer':int(delta),'delta_volume_m3':dm,'budget_m3':budget,
  'source_min_face_apex_bound':minq}

def load_npz(ref,label):
 if not isinstance(ref,dict) or not {'path','sha256'}<=set(ref):raise ValueError(label+' missing path/hash')
 p=Path(ref['path']).resolve()
 if sha(p)!=ref['sha256']:raise ValueError(label+' SHA mismatch')
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def coordmap(points):
 d={}
 for i,p in enumerate(np.asarray(points,np.float32)):
  k=fkey(p)
  if k in d:raise ValueError('duplicate Float32 coordinate key')
  d[k]=i
 return d
def map_rows_by_lineage(current_ref,reference_ref,current_data,rows,label,allow_collapsed=False):
 # Compose immediate parent->child point quotients from the exact typed-parent DAG.
 target_path=Path(reference_ref['path']).resolve()
 target_sha=str(reference_ref['sha256'])
 child_ref=dict(current_ref);child=current_data;maps=[];steps=[];seen=set()
 while True:
  cp=Path(child_ref['path']).resolve()
  if sha(cp)!=child_ref['sha256']:raise ValueError(label+' lineage node SHA mismatch')
  if cp==target_path:
   if child_ref['sha256']!=target_sha:raise ValueError(label+' target lineage hash mismatch')
   break
  if str(cp) in seen or len(seen)>128:raise ValueError(label+' parent cycle/depth')
  seen.add(str(cp))
  if 'typed_parent_candidate_path' not in child or 'typed_parent_candidate_sha256' not in child:
   raise ValueError(label+' lacks immediate typed-parent identity')
  pp=Path(str(np.asarray(child['typed_parent_candidate_path']).item())).resolve()
  ph=str(np.asarray(child['typed_parent_candidate_sha256']).item())
  if not pp.is_file() or sha(pp)!=ph:raise ValueError(label+' immediate parent pin invalid')
  if 'source_input_vertex_quotient' not in child:raise ValueError(label+' missing immediate point quotient')
  q=np.asarray(child['source_input_vertex_quotient'],np.int64)
  with np.load(pp,allow_pickle=False) as z: parent_n=len(z['points_world_m_f32'])
  if q.shape!=(parent_n,) or np.any(q<0) or np.any(q>=len(child['points_world_m_f32'])):
   raise ValueError(label+' immediate point quotient domain invalid')
  maps.append(q.copy())
  steps.append({'child':dict(child_ref),'parent':{'path':str(pp),'sha256':ph},'parent_point_rows':int(parent_n),'child_point_rows':int(len(child['points_world_m_f32'])),'point_quotient_sha256':hashlib.sha256(np.ascontiguousarray(q).view(np.uint8)).hexdigest()})
  child_ref={'path':str(pp),'sha256':ph}
  with np.load(pp,allow_pickle=False) as z:child={k:np.array(z[k],copy=True) for k in z.files}
 mapped=np.arange(len(child['points_world_m_f32']),dtype=np.int64)
 for q in reversed(maps):mapped=q[mapped]
 rr=np.asarray(rows,np.int64)
 if np.any(rr<0) or np.any(rr>=len(mapped)):raise ValueError(label+' row out of reference domain')
 out=mapped[rr]
 if not allow_collapsed and len(np.unique(out))!=len(out):raise ValueError(label+' protected rows were collapsed in current lineage')
 return out,{'reference':reference_ref,'current':current_ref,'descendant_steps':len(maps),
  'reference_row_domain':int(len(mapped)),'current_row_domain':int(len(current_data['points_world_m_f32'])),
  'protected_reference_rows':int(len(rr)),'protected_rows_unique':bool(len(np.unique(out))==len(out)), 'current_unique_rows':int(len(np.unique(out))), 'lineage_collapsed_rows':int(len(out)-len(np.unique(out))), 'steps_current_to_reference':steps}
def state_preflight(state_path,out_dir,role,*,cap=MOVE_CAP,seconds=90,max_ops=32,max_edge=.005,max_edges=100000):
 role=str(role)
 if role not in ('O','D'):raise ValueError('role must be O or D')
 if not 0<float(cap)<=MOVE_CAP:raise ValueError('ordinary O/D move cap must be <=0.5 mm')
 if not 0<float(max_edge)<=EDGE_CAP:raise ValueError('edge cap must be <=5 mm')
 state_path=Path(state_path).resolve();out=Path(out_dir).resolve();st=json.loads(state_path.read_text())
 for k in ('source_surfaces','Skin_current','Fat_current','exact_context'):
  if k not in st:raise ValueError('state missing '+k)
 if st.get('physical_admission') is not False or int(st.get('native_steps',-1))!=0:raise ValueError('state is not construction-only')
 proof=source_evidence(state_path,st)
 if not str(proof.get('status','')).startswith('PASS_'):raise ValueError('807 current source evidence failed')
 cr=st['exact_context'];ctxp=Path(cr['path']).resolve()
 if sha(ctxp)!=cr['sha256']:raise ValueError('exact context SHA mismatch')
 ctx=json.loads(ctxp.read_text())
 if not str(ctx.get('status','')).startswith('PASS_EXACT_'):raise ValueError('exact context status failed')
 if ctx.get('changed_vs_complete_retained_OID_both_walls_all185',{}).get('count')!=0 or ctx.get('changed_OID_joint_self_pairs',{}).get('count')!=0:raise ValueError('current O/I/D context has crossings')
 source_ref=st['source_surfaces'][role];surf=load_npz(source_ref,'current '+role)
 sp=np.asarray(surf['points_world_m_f32'],np.float32);sf=np.asarray(surf['faces'],np.int64)
 tissue='Skin' if role=='O' else 'Fat';vref=st[tissue+'_current'];va=load_npz(vref,'current '+tissue);cache=E.cache_volume(va)
 if not np.array_equal(T.boundary_identity(boundary(cache['t'])),T.boundary_identity(cache['desired'])):raise ValueError('input volume not on complete declared target')
 cross,cw=S.exact_used_row_crosswalk(sp,sf,cache['p'],tissue)
 scopep=Path(E.BACK_SCOPE_PATH).resolve()
 if sha(scopep)!=E.BACK_SCOPE_SHA:raise ValueError('back scope pin changed')
 scope=load_npz(pin(scopep),'back scope')
 scope_report_path=scopep.parent/'report.json';sr=json.loads(scope_report_path.read_text())
 if sr.get('status')!='PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT':raise ValueError('back-scope audit receipt status failed')
 if sr.get('protected_mask_asset',{}).get('sha256')!=sha(scopep):raise ValueError('back-scope report does not authenticate its mask asset')
 for r in ('O','I','D'):
  got=str(np.asarray(scope.get(r+'_candidate_sha256',np.asarray(['']))).reshape(-1)[0])
  if got!=sr['current_sources']['typed_roles'][r]['candidate']['sha256']:raise ValueError('back-scope candidate pin mismatch for '+r)
 ref=sr['current_sources']['typed_roles'][role]['candidate'];refdata=load_npz(ref,'back-scope reference '+role)
 back,back_lineage=map_rows_by_lineage(source_ref,ref,surf,scope[role+'_vertex_rows'],role+' actual back scope')
 expected={'O':177,'D':1673}[role]
 if len(back)!=expected:raise ValueError('actual backscope count differs')
 back_displacement=np.linalg.norm(sp[back].astype(np.float64)-refdata['points_world_m_f32'][np.asarray(scope[role+'_vertex_rows'],np.int64)].astype(np.float64),axis=1)
 if role=='O' and np.any(back_displacement!=0):raise ValueError('O accepted back geometry must remain bit-identical')
 actual_fat=np.empty(0,np.int64);i_back=np.empty(0,np.int64)
 if role=='D':
  ri=sr['current_sources']['typed_roles']['I']['candidate'];rida=load_npz(ri,'back-scope reference I')
  icref=st['source_surfaces']['I'];icur=load_npz(icref,'current I')
  i_back,i_lineage=map_rows_by_lineage(icref,ri,icur,scope['I_vertex_rows'],'I actual backscope',allow_collapsed=True)
  if len(i_back)!=790:raise ValueError('I backscope count differs from 790')
  ix,_=S.exact_used_row_crosswalk(icur['points_world_m_f32'],icur['faces'],cache['p'],'actual Fat back')
  current_fat=np.unique(ix[i_back])
  if np.any(current_fat<0) or len(current_fat)!=len(np.unique(i_back)):raise ValueError('actual current Fat back point map failed')
  fat_lut=coordmap(cache['p'])
  historical_fat=np.asarray([fat_lut.get(fkey(rida['points_world_m_f32'][int(r)]),-1)
      for r in np.asarray(scope['I_vertex_rows'],np.int64)],np.int64)
  historical_fat=historical_fat[historical_fat>=0]
  actual_fat=np.union1d(current_fat,historical_fat)
  if len(actual_fat)!=790:raise ValueError('current plus historical exact Fat back point union must contain 790 unique nodes')
 routes_obj=json.loads(A.PROT.read_text());routes=[]
 cmap=coordmap(sp)
 for rec in routes_obj.get('surface_routes',[]):
  if rec.get('role')==role:
   key=fkey(rec['target_world_m_f32'])
   if key not in cmap:raise ValueError('registered route missing')
   routes.append(cmap[key])
 routes=np.asarray(sorted(set(routes)),np.int64)
 es=np.sort(np.vstack((sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]])),axis=1);ue,ec=np.unique(es,axis=0,return_counts=True)
 rim=np.unique(ue[ec==1].ravel()).astype(np.int64);interior=ue[ec==2]
 protected=np.unique(np.r_[back,rim,routes]).astype(np.int64);pset=set(map(int,protected));bfset=set(map(int,actual_fat))
 det,q=E.measurements(cache['p'],cache['t']);bad=~E.M.passes(det,q)
 edge_lookup={tuple(map(int,e)):i for i,e in enumerate(interior)}
 inv=np.full(len(cache['p']),-1,np.int64);used=np.unique(sf.ravel());inv[cross[used]]=used
 score=Counter();bt=cache['t'][bad]
 for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
  p2=bt[:,(a,b)];u=inv[p2[:,0]];w=inv[p2[:,1]];ok=(u>=0)&(w>=0)&(u!=w)
  for x,y in np.sort(np.column_stack((u[ok],w[ok])),axis=1):
   i=edge_lookup.get((int(x),int(y)))
   if i is not None:score[i]+=1
 leng=np.linalg.norm(sp[interior[:,0]].astype(np.float64)-sp[interior[:,1]].astype(np.float64),axis=1)
 cand=[]
 for i,n in score.items():
  if n<=0 or leng[i]>max_edge:continue
  k,d=map(int,interior[i]);frows=np.flatnonzero(np.any((sf==k)|(sf==d),axis=1));verts=set(map(int,sf[frows].ravel()))
  if verts&pset:continue
  u,vn=cross[k],cross[d]
  if u<0 or vn<0:continue
  star=np.union1d(np.flatnonzero(np.any(cache['t']==u,axis=1)),np.flatnonzero(np.any(cache['t']==vn,axis=1)))
  if set(map(int,np.unique(cache['t'][star].ravel())))&bfset:continue
  cand.append((int(i),k,d,int(n),float(leng[i])))
 cand.sort(key=lambda z:(-z[3],z[4],z[1],z[2]));cand=cand[:int(max_edges)]
 und=np.asarray([[k,d] for _,k,d,_,_ in cand],np.int64).reshape(-1,2)
 orient=np.asarray([(k,d) for _,k,d,_,_ in cand]+[(d,k) for _,k,d,_,_ in cand],np.int64).reshape(-1,2)
 out.mkdir(parents=True,exist_ok=False);ins=out/'inputs';ins.mkdir()
 edgep=ins/('candidate-'+role+'-edges.npz')
 np.savez_compressed(edgep,candidate_edges=orient,undirected_edges=und,
  edge_length_m=np.asarray([z[4] for z in cand]),bad_tet_incidence=np.asarray([z[3] for z in cand]))
 protp=ins/('protected-'+role+'-rows.npz')
 np.savez_compressed(protp,protected_surface_rows=protected,back_scope_rows=back,
  back_scope_reference_rows=np.asarray(scope[role+'_vertex_rows'],np.int64),back_scope_current_rows=back,
  source_rim_rows=rim,route_anchor_rows=routes,actual_back_fat_volume_point_rows=actual_fat,
  actual_back_fat_current_rows=current_fat if role=='D' else np.empty(0,np.int64),
  actual_back_fat_historical_rows=historical_fat if role=='D' else np.empty(0,np.int64),
  actual_back_i_surface_rows=i_back,actual_back_i_reference_rows=np.asarray(scope['I_vertex_rows'],np.int64) if role=='D' else np.empty(0,np.int64),actual_back_fat_reference_rows=historical_fat if role=='D' else np.empty(0,np.int64))
 cfg={'schema':'h706.coupled-anchor-collapse-OD-run-config.v1','role':role,'volume_role':tissue,
  'surface':source_ref,'baseline_surfaces':st['source_surfaces'],'volume':vref,
  'candidate_edges':pin(edgep),'candidate_edges_key':'candidate_edges','protection':pin(protp),
  'source_state':pin(state_path),'current_exact_context':pin(ctxp),'move_cap_m':float(cap),
  'producer':pin(__file__),'base_791':pin(BASE),'base_781':pin(E.__file__),
  'source_evidence_owner':pin(SOURCE807),
  'max_operations':int(max_ops),'time_limit_s':float(seconds)}
 cfgp=ins/'run-config.json';E._write_json(cfgp,cfg)
 prep={'schema':'h706.coupled-anchor-collapse-OD-preflight.v1','status':'PASS_READ_ONLY_PREFLIGHT_NO_SOLVER',
  'role':role,'volume_role':tissue,'state':pin(state_path),'current_exact_context':pin(ctxp),
  'source_evidence_status':proof.get('status'),'surface_parent':source_ref,'volume_parent':vref,
  'back_scope':pin(scopep),'back_scope_audit_report':pin(scope_report_path),'back_scope_reference':ref,
  'preserved_back_vertices':{'O_scope_rows':177,'D_scope_rows':1673,'historical_I_back_scope_rows':790 if role=='D' else 0, 'current_unique_I_back_rows':int(len(np.unique(i_back))) if role=='D' else 0, 'current_Fat_back_nodes':int(len(current_fat)) if role=='D' else 0, 'historical_exact_Fat_back_nodes':int(len(historical_fat)) if role=='D' else 0, 'protected_current_plus_historical_Fat_back_nodes':int(len(actual_fat))},
  'back_row_lineage':back_lineage,'actual_back_I_row_lineage':i_lineage if role=='D' else None,'back_current_displacement_m':{'max':float(np.max(back_displacement)) if len(back_displacement) else 0.0,
   'policy':'O exact coordinate identity; D uses row-preserving typed lineage and leaves moved current coordinates as the current geometry'},
  'protected_surface_rows':len(protected),'rim_vertices':len(rim),'registered_route_rows':len(routes),
  'actual_back_Fat_volume_rows':len(actual_fat),'native_bad_before':int(np.count_nonzero(bad)),
  'surface_interior_edges':len(interior),'bad_incident_edge_count_before_protection':len(score),
  'scheduled_undirected_edges':len(cand),'scheduled_orientations':len(orient),
  'edge_roster':pin(edgep),'protection':pin(protp),'run_config':pin(cfgp),
  'move_cap_m':float(cap),'max_edge_length_m':float(max_edge),
  'producer':pin(__file__),'base_791':pin(BASE),'base_781':pin(E.__file__),
  'source_evidence_owner':pin(SOURCE807),
  'volume_policy':'single-role exact signed delta; unchanged 1% local/1e-4 regional budgets; mass reconciliation pending',
  'back_policy':'current accepted back vertices remain fixed; D full Fat star excludes all 790 mapped actual back Fat points',
  'native_q_gate_unchanged':NATIVE_Q,'minimum_cell_volume_m3_unchanged':MIN_CELL_VOLUME_M3,
  'physical_admission':False,'native_steps':0,'mass_reconciliation':'PENDING'}
 E._write_json(ins/'state-preparation.json',prep)
 return cfgp,prep

def validate_context_binding(st,cfg):
 ref=cfg.get('current_exact_context')
 state_ref=st.get('exact_context')
 if not isinstance(ref,dict) or not isinstance(state_ref,dict):raise ValueError('missing exact context reference')
 if Path(ref.get('path','')).resolve()!=Path(state_ref.get('path','')).resolve() or ref.get('sha256')!=state_ref.get('sha256'):
  raise ValueError('run config context differs from frozen state')
 cp=Path(ref['path']).resolve()
 if sha(cp)!=ref['sha256']:raise ValueError('exact context report SHA mismatch')
 report=json.loads(cp.read_text())
 if not str(report.get('status','')).startswith('PASS_EXACT_'):raise ValueError('exact context report status failed')
 if report.get('physical_admission') is not False or int(report.get('native_steps',-1))!=0:
  raise ValueError('exact context carries unexpected admission state')
 for key in ('changed_vs_complete_retained_OID_both_walls_all185','changed_OID_joint_self_pairs'):
  if report.get(key,{}).get('count')!=0:raise ValueError('exact context has nonzero crossing count: '+key)
 roles=report.get('roles',{})
 source=st.get('source_surfaces',{})
 baseline=cfg.get('baseline_surfaces',{})
 for role in ('O','I','D'):
  cand=roles.get(role,{}).get('candidate',{})
  sr=source.get(role,{})
  br=baseline.get(role,{})
  if not cand or not sr or Path(cand.get('path','')).resolve()!=Path(sr.get('path','')).resolve() or cand.get('sha256')!=sr.get('sha256'):
   raise ValueError('exact context candidate does not bind current state role '+role)
  if Path(br.get('path','')).resolve()!=Path(sr.get('path','')).resolve() or br.get('sha256')!=sr.get('sha256'):
   raise ValueError('run baseline differs from frozen state role '+role)
 cr=report.get('config')
 if isinstance(cr,dict):
  cp2=Path(cr.get('path','')).resolve()
  if not cp2.is_file() or sha(cp2)!=cr.get('sha256'):raise ValueError('exact context source config pin invalid')
  cc=json.loads(cp2.read_text()).get('candidate_surfaces',{})
  for role in ('O','I','D'):
   rr=cc.get(role,{})
   cand=roles[role]['candidate']
   if Path(rr.get('path','')).resolve()!=Path(cand['path']).resolve() or rr.get('sha256')!=cand.get('sha256'):
    raise ValueError('exact context source config candidate mismatch '+role)
 return report

def load_config(path):
 p=Path(path).resolve();cfg=json.loads(p.read_text())
 if cfg.get('schema')!='h706.coupled-anchor-collapse-OD-run-config.v1':raise ValueError('bad config schema')
 if pinned(cfg.get('producer',{}),'794 producer')!=Path(__file__).resolve():raise ValueError('config producer pin mismatch')
 if pinned(cfg.get('base_791',{}),'791 parent')!=BASE or pinned(cfg.get('base_781',{}),'781 engine')!=Path(E.__file__).resolve():raise ValueError('frozen owner parent pin mismatch')
 if pinned(cfg.get('source_evidence_owner',{}),'807 source verifier')!=SOURCE807:raise ValueError('807 source verifier pin mismatch')
 role=cfg.get('role');tissue=cfg.get('volume_role')
 if (role,tissue) not in (('O','Skin'),('D','Fat')):raise ValueError('bad role/tissue pairing')
 surf=load_npz(cfg['surface'],'source surface');vol=load_npz(cfg['volume'],'source volume')
 edge=np.asarray(load_npz(cfg['candidate_edges'],'edge roster')[cfg['candidate_edges_key']],np.int64)
 prot=load_npz(cfg['protection'],'protection roster')
 state_path=Path(cfg['source_state']['path']).resolve()
 st=json.loads(state_path.read_text())
 if sha(state_path)!=cfg['source_state']['sha256']:raise ValueError('state pin mismatch')
 proof=source_evidence(state_path,st)
 if not str(proof.get('status','')).startswith('PASS_'):raise ValueError('current 807 source evidence failed on run load')
 if st['source_surfaces'][role]['sha256']!=cfg['surface']['sha256'] or st[tissue+'_current']['sha256']!=cfg['volume']['sha256']:raise ValueError('parent pin mismatch')
 if st.get('exact_context',{}).get('sha256')!=cfg.get('current_exact_context',{}).get('sha256'):raise ValueError('current exact-context pin mismatch')
 validate_context_binding(st,cfg)
 if not (0<float(cfg['move_cap_m'])<=MOVE_CAP):raise ValueError('ordinary O/D movement cap invalid')
 state_surface=load_npz(st['source_surfaces'][role],'state source')
 if not np.array_equal(surf['faces'],state_surface['faces']) or not np.array_equal(surf['points_world_m_f32'],state_surface['points_world_m_f32']):raise ValueError('surface parent geometry mismatch')
 if edge.ndim!=2 or edge.shape[1]!=2:raise ValueError('edge roster shape invalid')
 if not np.array_equal(T.boundary_identity(boundary(vol['tetrahedra'])),T.boundary_identity(vol['desired_boundary_faces'])):raise ValueError('volume target baseline mismatch')
 cache=E.cache_volume(vol)
 protected_fat=np.asarray(prot['actual_back_fat_volume_point_rows'],np.int64)
 if np.any(protected_fat<0) or np.any(protected_fat>=len(cache['p'])) or len(np.unique(protected_fat))!=len(protected_fat):
  raise ValueError('actual back Fat volume protection map invalid')
 return {'path':p,'cfg':cfg,'surface':surf,'volume':cache,'edges':edge,'protection':prot}

def source_output(parent,ops,points,faces,faceparents,role,parent_ref):
 data={k:np.array(v,copy=True) for k,v in parent.items()};nface=len(parent['faces']);rowop={}
 unions=[];anc=[];opsby=[]
 for oi,op in enumerate(ops):
  for r in map(int,op['surface_full_endpoint_face_rows']):
   if r in rowop:raise ValueError('overlapping surface stars')
   rowop[r]=oi
 cache={}
 for oi,op in enumerate(ops):
  rr=op['surface_full_endpoint_face_rows'];cache[oi]=(B.union_tuples(parent,rr),B.ancestry_union(parent,rr))
 for r in map(int,faceparents):
  oi=rowop.get(r,-1)
  if oi>=0:u,a=cache[oi];unions.append(u);anc.append(a);opsby.append(oi)
  else:unions.append(A.rows_union(parent,r));anc.append(A.rows_anc(parent,r));opsby.append(-1)
 uo,ud=A.csr(unions,7);ao,ad=A.csr(anc,3)
 for k in list(data):
  if k.startswith('source_') and k.endswith('_operation_rows'):data.pop(k)
 data['points_world_m_f32']=np.asarray(points,np.float32)
 for k,v in list(data.items()):
  a=np.asarray(v)
  if k!='points_world_m_f32' and ('points_world_m' in k or k=='points') and a.shape==np.asarray(points).shape and np.issubdtype(a.dtype,np.floating):data[k]=np.asarray(points,dtype=a.dtype)
 data['faces']=np.asarray(faces,np.int64);data['source_input_face_rows']=np.asarray(faceparents,np.int64)
 fid=np.asarray(parent.get('quality_star_face_output_to_input_row',np.arange(nface)),np.int64)[faceparents].copy()
 opid=np.asarray(opsby,np.int64);fid[opid>=0]=-1;data['quality_star_face_output_to_input_row']=fid
 data['quality_star_patch_union_offsets']=uo;data['face_ancestry_offsets']=ao
 data['source_collapse_operation_rows']=opid
 q=np.arange(len(points),dtype=np.int64)
 for op in ops:q[int(op['drop'])]=int(op['keep'])
 data['source_input_vertex_quotient']=q
 for j,k in enumerate(A.T):data[k]=np.asarray([u[0][j] for u in unions],np.int64)
 for j,k in enumerate(A.U):data[k]=ud[:,j]
 for j,k in enumerate(A.A):data[k]=ad[:,j]
 data['coupled_anchor_operations_json']=np.asarray([json.dumps({
  'role':role,'keep_parent_point_row':int(o['keep']),'drop_parent_point_row':int(o['drop']),
  'moved_anchor_world_m':np.asarray(o['anchor_f32'],np.float32).tolist(),
  'move_cap_m':float(o['fit']['move_cap_m']),
  'complete_parent_face_rows':np.asarray(o['surface_full_endpoint_face_rows'],np.int64).tolist(),
  'retired_parent_face_rows':np.asarray(o['surface_retired_face_rows'],np.int64).tolist(),
  'physical_admission':False,'native_steps':0},sort_keys=True) for o in ops])
 data['typed_parent_candidate_path']=np.asarray(parent_ref['path'])
 data['typed_parent_candidate_sha256']=np.asarray(parent_ref['sha256'])
 data['typed_parent_role']=np.asarray(role)
 return data

def build_volume(cache,ops):
 v=cache;p=v['p'].copy();qmap=np.arange(len(p),dtype=np.int64);retire=np.zeros(len(v['t']),bool)
 rt=[];rm=[];direct=[];union=[];offs=[0]
 for op in ops:
  u,w=op['volume_endpoints'];p[u]=op['anchor_f32'];qmap[w]=u;rows=op['volume_star_cells'];retire[rows]=True
  rt.append(op['replacement_tets']);rm.append(op['replacement_materials']);direct.append(op['replacement_parent_rows'])
  for _ in op['replacement_tets']:
   union.extend(rows.tolist());offs.append(len(union))
 keep=np.flatnonzero(~retire);replacement=np.concatenate(rt);mnew=np.concatenate(rm);drows=np.concatenate(direct)
 t=np.vstack((v['t'][keep],replacement));mat=np.r_[v['mat'][keep],mnew]
 if len({ckey(x) for x in t})!=len(t):raise ValueError('duplicate global cell')
 det,q=E.measurements(p,t)
 if np.any(det<=0) or not exact_pos(p,t):raise ValueError('global nonpositive')
 target=qmap[v['desired']];target=target[~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1)]
 bd=boundary(t)
 if not np.array_equal(T.boundary_identity(bd),T.boundary_identity(target)):raise ValueError('global target boundary mismatch')
 flux=exact_sum(p,t);oldflux=exact_sum(v['p'],v['t']);deltas=sum(int(o['exact_delta_integer']) for o in ops)
 if flux-oldflux!=deltas:raise ValueError('exact global flux delta mismatch')
 imap=np.full(len(v['t']),-1,np.int64);imap[keep]=np.arange(len(keep))
 a={k:np.array(x,copy=True) for k,x in v['arrays'].items()}
 for k,x in list(a.items()):
  z=np.asarray(x)
  if k!='points_world_m' and ('points_world_m' in k or k=='points') and z.shape==p.shape and np.issubdtype(z.dtype,np.floating):a[k]=p.astype(z.dtype)
 a.update(points_world_m=p,tetrahedra=t,material_indices=mat,boundary_faces=bd,desired_boundary_faces=target,
  output_cell_to_input_cell_rows=np.r_[keep,drows],input_cell_to_output_cell_rows=imap,retained_input_cell_rows=keep,
  replacement_parent_cell_rows=np.asarray(union,np.int64),replacement_parent_offsets=np.asarray(offs,np.int64),
  replacement_tetrahedra=replacement,input_node_to_output_node_rows=qmap,
  output_node_to_input_node_rows=np.arange(len(p),dtype=np.int64),output_point_to_input_point_rows=np.arange(len(p),dtype=np.int64),
  output_point_parent_offsets=np.zeros(len(p)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
 bad=int(np.count_nonzero(~E.M.passes(det,q)));before=int(np.count_nonzero(v['bad']))
 if bad>before:raise ValueError('global quality regression')
 return a,{'cells_before':len(v['t']),'cells_after':len(t),'native_bad_before':before,
  'native_bad_after':bad,'native_bad_removed':before-bad,'exact_volume_delta_integer':str(flux-oldflux),
  'exact_volume_delta_m3':(flux-oldflux)*DET_TO_M3,'nonpositive':0,'target_boundary_exact':True}

def make_context(run_dir,x,report,path):
 role=x['cfg']['role'];candidate=pin(Path(run_dir)/('candidate-'+role+'-typed-current.npz'))
 base=x['cfg']['baseline_surfaces'];cand={r:dict(base[r]) for r in ('O','I','D')};cand[role]=candidate
 ctx={'schema':'h706.combined-current-exact-surface-diff-config.v1','baseline_candidates':base,
  'candidate_surfaces':cand,'changed_roles':[role],
  'bounded_point_repair':{role:{'maximum_displacement_m':float(x['cfg']['move_cap_m'])}},
  'source_candidate_run':{'operation_count':int(report['accepted_operations']),'surface_output':candidate,'role':role},
  'physical_admission':False,'native_steps':0,'mass_reconciliation':'PENDING_SINGLE_ROLE_VOLUME_CHANGE'}
 E._write_json(path,ctx);return pin(path)

def run_batch(config_path,out_dir):
 x=load_config(config_path);cfg=x['cfg'];role=cfg['role'];tissue=cfg['volume_role'];out=Path(out_dir).resolve()
 out.mkdir(parents=True,exist_ok=False);ops=[];reject=Counter();useds=set();usedp=set();usedc=set();start=time.monotonic()
 prot=x['protection']
 for k,d in x['edges']:
  if time.monotonic()-start>=float(cfg['time_limit_s']) or len(ops)>=int(cfg['max_operations']):break
  op=prep_operation(x['surface'],x['volume'],role,k,d,prot['protected_surface_rows'],
   prot['route_anchor_rows'],prot['actual_back_fat_volume_point_rows'],float(cfg['move_cap_m']))
  if op['status']!='PASS':reject[op['status']]+=1;continue
  ss=set(map(int,op['surface_star_vertices']));pp=set(map(int,op['volume_star_points']));cc=set(map(int,op['volume_star_cells']))
  if ss&useds or pp&usedp or cc&usedc:reject['overlapping_complete_stars']+=1;continue
  ops.append(op);useds|=ss;usedp|=pp;usedc|=cc
 if not ops:
  report={'schema':'h706.coupled-anchor-collapse-OD-state.v1','status':'NO_VALID_SINGLE_ROLE_OPS',
   'role':role,'volume_role':tissue,'accepted_operations':0,'rejections':dict(reject),
   'physical_admission':False,'native_steps':0}
  E._write_json(out/'state.json',report);return report
 sp=np.asarray(x['surface']['points_world_m_f32'],np.float32).copy();qmap=np.arange(len(sp),dtype=np.int64)
 for o in ops:qmap[o['drop']]=o['keep'];sp[o['keep']]=o['anchor_f32']
 basef=np.asarray(x['surface']['faces'],np.int64);fm=qmap[basef];valid=~np.any(np.diff(np.sort(fm,axis=1),axis=1)==0,axis=1);parents=np.flatnonzero(valid);faces=fm[valid]
 source=source_output(x['surface'],ops,sp,faces,parents,role,cfg['surface'])
 sourcep=out/('candidate-'+role+'-typed-current.npz');np.savez_compressed(sourcep,**source)
 vol,vr=build_volume(x['volume'],ops);volp=out/(tissue.lower()+'-'+role.lower()+'-volume-proposal.npz');np.savez_compressed(volp,**vol)
 records=[]
 for o in ops:records.append({'keep':int(o['keep']),'drop':int(o['drop']),'anchor_f32':np.asarray(o['anchor_f32'],np.float32).tolist(),
  'surface_parent_faces':np.asarray(o['surface_full_endpoint_face_rows'],np.int64).tolist(),
  'surface_retired_faces':np.asarray(o['surface_retired_face_rows'],np.int64).tolist(),
  'surface_star_vertices':np.asarray(o['surface_star_vertices'],np.int64).tolist(),
  'volume_endpoints':list(map(int,o['volume_endpoints'])),'full_star_cells':np.asarray(o['volume_star_cells'],np.int64).tolist(),
  'full_star_points':np.asarray(o['volume_star_points'],np.int64).tolist(),
  'replacement_tets':np.asarray(o['replacement_tets'],np.int64).tolist(),
  'replacement_parent_rows':np.asarray(o['replacement_parent_rows'],np.int64).tolist(),
  'replacement_materials':np.asarray(o['replacement_materials'],np.int32).tolist(),
  'old_bad':int(o['old_bad']),'new_bad':int(o['new_bad']),'exact_delta_integer':str(o['exact_delta_integer']),
  'exact_delta_volume_m3':float(o['delta_volume_m3']),'budget_m3':float(o['budget_m3'])})
 report={'schema':'h706.coupled-anchor-collapse-OD-state.v1','status':'CANDIDATE_PENDING_775_697_217',
  'role':role,'volume_role':tissue,'accepted_operations':len(ops),'rejections':dict(reject),'operation_records':records,
  'source_parent':cfg['surface'],'volume_parent':cfg['volume'],'output_source':pin(sourcep),'output_volume':pin(volp),'run_config':pin(config_path),'producer':pin(__file__),'base_791':pin(BASE),'base_781':pin(E.__file__),
  'protected_rows':cfg['protection'],'current_exact_context':cfg['current_exact_context'],'source_state':cfg['source_state'],
  'volume_quality':vr,'exact_volume_delta_integer':vr['exact_volume_delta_integer'],
  'exact_volume_delta_m3':vr['exact_volume_delta_m3'],'native_q_gate_unchanged':NATIVE_Q,
  'minimum_cell_volume_m3_unchanged':MIN_CELL_VOLUME_M3,
  'source_normal_apex_and_link_checks':'PASS_LOCAL','complete_target_boundary':'PASS_EXACT',
  'independent_replay':'REQUIRED','all185_775':'REQUIRED','actual_697_217':'REQUIRED',
  'physical_admission':False,'native_steps':0,'mass_reconciliation':'PENDING_SINGLE_ROLE_VOLUME_CHANGE'}
 report['775_context_config']=make_context(out,x,report,out/'775-context-config.json')
 E._write_json(out/'state.json',report);return report

def input_map(n,keep):
 a=np.full(int(n),-1,np.int64);a[np.asarray(keep,np.int64)]=np.arange(len(keep),dtype=np.int64);return a

def independent_replay(config_path,run_dir,out_path):
 x=load_config(config_path);cfg=x['cfg'];role=cfg['role'];tissue=cfg['volume_role'];rd=Path(run_dir).resolve()
 report=json.loads((rd/'state.json').read_text())
 if report.get('status')!='CANDIDATE_PENDING_775_697_217':raise ValueError('candidate status is not replayable')
 if report.get('run_config',{}).get('sha256')!=sha(config_path):raise ValueError('run config pin mismatch')
 if pinned(report.get('producer',{}),'794 candidate producer')!=Path(__file__).resolve():raise ValueError('candidate producer pin mismatch')
 surf=x['surface'];points=np.asarray(surf['points_world_m_f32'],np.float32);faces0=np.asarray(surf['faces'],np.int64)
 ops=report['operation_records'];q=np.arange(len(points),dtype=np.int64);sp=points.copy();faceop={}
 for oi,o in enumerate(ops):
  k,d=int(o['keep']),int(o['drop'])
  if q[k]!=k or q[d]!=d:raise ValueError('overlapping source quotient')
  star=np.union1d(np.flatnonzero(np.any(faces0==k,axis=1)),np.flatnonzero(np.any(faces0==d,axis=1)))
  if not np.array_equal(star,np.asarray(o['surface_parent_faces'],np.int64)):raise ValueError('source face-star mismatch')
  for r in star:
   if int(r) in faceop:raise ValueError('overlapping source face stars')
   faceop[int(r)]=oi
  q[d]=k;sp[k]=np.asarray(o['anchor_f32'],np.float32)
 fm=q[faces0];valid=~np.any(np.diff(np.sort(fm,axis=1),axis=1)==0,axis=1);fparents=np.flatnonzero(valid);fout=fm[valid]
 expected_source=source_output(surf,ops,sp,fout,fparents,role,cfg['surface'])
 sout=load_npz(report['output_source'],'typed source output')
 for key in ('points_world_m_f32','faces','source_input_face_rows','source_input_vertex_quotient','source_collapse_operation_rows','quality_star_patch_union_offsets','face_ancestry_offsets','quality_star_face_output_to_input_row',*list(A.T),*list(A.U),*list(A.A)):
  if key not in sout or not np.array_equal(sout[key],expected_source[key]):raise ValueError('source/ancestry readback mismatch '+key)
 for key in surf:
  a=np.asarray(surf[key])
  if key!='points_world_m_f32' and ('points_world_m' in key or key=='points') and a.shape==sp.shape and np.issubdtype(a.dtype,np.floating) and key in sout:
   expected=a.copy()
   for o in ops:expected[int(o['keep'])]=np.asarray(o['anchor_f32'],dtype=a.dtype)
   if not np.array_equal(sout[key],expected):raise ValueError('source coordinate alias mismatch '+key)
 if str(np.asarray(sout['typed_parent_candidate_sha256']).item())!=cfg['surface']['sha256']:raise ValueError('typed parent pin mismatch')
 prot=x['protection'];fixed=np.asarray(prot['protected_surface_rows'],np.int64)
 if not np.array_equal(sout['points_world_m_f32'][fixed].view(np.uint32),points[fixed].view(np.uint32)):raise ValueError('protected source back/rim/route point changed')
 for o in ops:
  parent_rows=np.asarray(o['surface_parent_faces'],np.int64)
  kept=parent_rows[~np.any(np.diff(np.sort(q[faces0[parent_rows]],axis=1),axis=1)==0,axis=1)]
  newrows=q[faces0[kept]]
  oldn=normals(points,faces0[kept]);newn=normals(sp,newrows)
  den=np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1)
  if np.any(den<=0) or np.any(np.einsum('ij,ij->i',oldn,newn)<.9*den):raise ValueError('source normal replay failed')
  qok,_=source_q(sp,newrows)
  if not qok:raise ValueError('source face-apex replay failed')
 v=x['volume'];bt=v['t'];bm=v['mat'];bp=v['p'];p=bp.copy();qv=np.arange(len(p),dtype=np.int64);retire=np.zeros(len(bt),bool)
 news=[];newm=[];parents=[];unions=[];offs=[0];deltas=[]
 for o in ops:
  u,w=map(int,o['volume_endpoints']);star=np.flatnonzero(np.any((bt==u)|(bt==w),axis=1))
  if not np.array_equal(star,np.asarray(o['full_star_cells'],np.int64)):raise ValueError('volume cell-star mismatch')
  if not volume_link(bt,u,w):raise ValueError('volume link replay failed')
  p[u]=np.asarray(o['anchor_f32'],np.float32);qv[w]=u;retire[star]=True;old=bt[star];mapped=np.where(old==w,u,old)
  dead=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1);nt=mapped[~dead];pr=star[~dead];m=bm[pr]
  if not np.array_equal(nt,np.asarray(o['replacement_tets'],np.int64)) or not np.array_equal(pr,np.asarray(o['replacement_parent_rows'],np.int64)) or not np.array_equal(m,np.asarray(o['replacement_materials'],np.int32)):raise ValueError('replacement lineage mismatch')
  de=exact_sum(p,nt)-exact_sum(bp,old)
  if str(de)!=str(o['exact_delta_integer']):raise ValueError('exact local volume delta mismatch')
  if abs(de*DET_TO_M3)>float(o['budget_m3']):raise ValueError('volume budget mismatch')
  nd,nq=E.measurements(p,nt);od,oq=E.measurements(bp,old)
  if np.any(nd<=0) or not exact_pos(p,nt) or not np.all(E.M.passes(nd,nq)):raise ValueError('replacement native/positive gate')
  oldbad=int(np.count_nonzero(~E.M.passes(od,oq)));newbad=int(np.count_nonzero(~E.M.passes(nd,nq)))
  if oldbad!=int(o['old_bad']) or newbad!=int(o['new_bad']) or oldbad<=newbad:raise ValueError('strict quality gain mismatch')
  for _ in range(len(nt)):unions.extend(star.tolist());offs.append(len(unions))
  news.append(nt);newm.append(m);parents.append(pr);deltas.append(int(de))
 keep=np.flatnonzero(~retire);rt=np.concatenate(news);rm=np.concatenate(newm);rp=np.concatenate(parents)
 t=np.vstack((bt[keep],rt));mat=np.r_[bm[keep],rm];target=qv[v['desired']];target=target[~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1)];bd=boundary(t)
 if not np.array_equal(T.boundary_identity(bd),T.boundary_identity(target)):raise ValueError('complete boundary mismatch')
 vo=load_npz(report['output_volume'],'volume output')
 if not np.array_equal(vo['points_world_m'],p) or not np.array_equal(vo['tetrahedra'],t) or not np.array_equal(vo['material_indices'],mat):raise ValueError('volume arrays mismatch')
 backfat=np.asarray(x['protection']['actual_back_fat_volume_point_rows'],np.int64)
 if len(backfat) and not np.array_equal(vo['points_world_m'][backfat].view(np.uint32),bp[backfat].view(np.uint32)):raise ValueError('protected actual Fat back node moved')
 checks={'boundary_faces':bd,'desired_boundary_faces':target,'retained_input_cell_rows':keep,
  'output_cell_to_input_cell_rows':np.r_[keep,rp],'input_cell_to_output_cell_rows':input_map(len(bt),keep),
  'replacement_parent_cell_rows':np.asarray(unions,np.int64),'replacement_parent_offsets':np.asarray(offs,np.int64),
  'replacement_tetrahedra':rt,'input_node_to_output_node_rows':qv}
 for k,want in checks.items():
  if k not in vo or not np.array_equal(vo[k],want):raise ValueError('volume lineage mismatch '+k)
 det,qall=E.measurements(p,t)
 if np.any(det<=0) or not exact_pos(p,t):raise ValueError('global nonpositive')
 before=int(np.count_nonzero(v['bad']));after=int(np.count_nonzero(~E.M.passes(det,qall)))
 if after>before:raise ValueError('global native quality regression')
 delta=exact_sum(p,t)-exact_sum(bp,bt)
 if delta!=sum(deltas):raise ValueError('global exact flux mismatch')
 result={'schema':'h706.coupled-anchor-collapse-OD-replay.v1','status':'PASS_INDEPENDENT_RECONSTRUCTION',
  'role':role,'volume_role':tissue,'operation_count':len(ops),'native_bad_before':before,
  'native_bad_after':after,'native_bad_removed':before-after,'nonpositive':0,
  'complete_target_boundary':True,'typed_parent_union_replay':True,
  'exact_volume_delta_integer':str(delta),'exact_volume_delta_m3':delta*DET_TO_M3,
  'mass_reconciliation':'PENDING_SINGLE_ROLE_VOLUME_CHANGE','physical_admission':False,'native_steps':0}
 E._write_json(out_path,result);return result

def fixture():
 x=np.asarray([0,0,0],np.float32);tr=np.asarray([[[1,0,0],[0,1,0],[0,0,1]]],np.float32)
 p=np.vstack((x,tr.reshape(-1,3)));t=np.array([[0,1,2,3]],np.int64);det,q=E.measurements(p,t)
 face=p[t[0,1:]][None,:,:]
 r=fit_single('Skin',x,face,float(det.sum()),exact_sum(p,t),float(det.sum()),float(det.sum()),cap=MOVE_CAP)
 if r.get('status')!='PASS' or r['minimum_new_native_q']<NATIVE_Q:raise AssertionError('single-role fitter control failed')
 pneg=np.vstack((np.asarray([0,0,2],np.float32),face.reshape(-1,3))).astype(np.float32);dneg,qneg=E.measurements(pneg,np.array([[0,1,2,3]],np.int64))
 if dneg[0]>0 and exact_pos(pneg,np.array([[0,1,2,3]],np.int64)):raise AssertionError('negative determinant control passed')
 return {'schema':'h706.coupled-anchor-collapse-OD-fixture.v1','status':'PASS',
  'single_role_fit':'PASS','native_q_gate_unchanged':NATIVE_Q,
  'minimum_cell_volume_m3_unchanged':MIN_CELL_VOLUME_M3,'negative_volume_rejected':True,
  'paired_zero_delta_constraint':'NOT_APPLIED','ordinary_move_cap_m':MOVE_CAP,
  'physical_admission':False,'native_steps':0}

def main():
 ap=argparse.ArgumentParser(description='792 one-role O/D optimized-anchor collapse')
 ap.add_argument('--mode',choices=('fixture','prepare-state','run','replay'),default='fixture')
 ap.add_argument('--role',choices=('O','D'));ap.add_argument('--state',type=Path);ap.add_argument('--config',type=Path)
 ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--move-cap-m',type=float,default=MOVE_CAP);ap.add_argument('--max-operations',type=int,default=32)
 ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--max-edge-length-m',type=float,default=EDGE_CAP)
 ap.add_argument('--max-edges',type=int,default=100000);a=ap.parse_args()
 if a.mode=='fixture':
  r=fixture();E._write_json(a.output,r);print(json.dumps(r,indent=2));return
 if a.mode=='prepare-state':
  if a.state is None or a.role is None:ap.error('prepare-state requires --state and --role')
  cfg,prep=state_preflight(a.state,a.output,a.role,cap=a.move_cap_m,seconds=a.seconds,
   max_ops=a.max_operations,max_edge=a.max_edge_length_m,max_edges=a.max_edges)
  print(json.dumps({'status':prep['status'],'preparation':pin(a.output/'inputs'/'state-preparation.json'),
   'run_config':pin(cfg),'candidate_solver':'NOT_STARTED','physical_admission':False,'native_steps':0},indent=2));return
 if a.mode=='run':
  if a.config is None:ap.error('run requires --config')
  r=run_batch(a.config,a.output);print(json.dumps({k:v for k,v in r.items() if k!='operation_records'},indent=2));return
 if a.mode=='replay':
  if a.config is None or a.run_dir is None:ap.error('replay requires --config --run-dir')
  r=independent_replay(a.config,a.run_dir,a.output);print(json.dumps(r,indent=2));return

if __name__=='__main__':main()

