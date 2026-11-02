#!/usr/bin/env python3
"""Current-1020 direct-mask cut-volume sensitivity and bounded I-step candidate."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
STATE=H/'all-quality-1020-zero-skin-fat-source-qualified-root/verified-state.json'
STATE_SHA='11044a88b2d45ab9a9625f9a2cac46b8a952032b9ac0a07274710c316f5a88a9'
TYPED=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
TYPED_SHA='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'
I_PATH=H/'all-quality-1009-point-update-1018-root/actual-001/candidate-I-point-updated.npz'
O_PATH=H/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
D_PATH=H/'all-quality-958-D014-root/proposal-003/candidate-D-typed-current.npz'
SKIN_PATH=H/'all-quality-1009-point-update-1018-root/actual-001/skin-volume-point-updated.npz'
FAT_PATH=H/'all-quality-1009-point-update-1018-root/actual-001/fat-volume-point-updated.npz'
PINS={'I':'5557cbb88b46ea83d7977d60e1d362a10c48c290b77afb39b273efa29d32e323','O':'10169962e298586d63794bf94515ecc3a9b1ce91e66cc60566a6919d7c856ac3','D':'63cec05102fd2dc670d9220ec6472cd7d9ffa7085b4694719c605947086c7da9','Skin':'41291a7298517ab250a64e4bbda6bbee7591b7d2b594aacc1dce6d6fc78025ba','Fat':'b83df0f4a4ff183bd147f0c0d6236a22e3ea6ce1ae9e9531ffc1c457c2ffac88'}
CENSUS_DIR=H/'current-fractional-semantic-native-1047/actual-027-proof-slot-closure/actual-027-full-deficit-component-cap-009'
CENSUS=CENSUS_DIR/'component-census/component-census.json'; FULL=CENSUS_DIR/'full-unreachable-target-rows.csv'; SUMMARY=CENSUS_DIR/'full-unreachable-component-summary.json'
LABELS=H/'current-native-1024-root/current-433-1044/volume-export/volume/child.labels.csv'; LABELS_SHA='c48c944a0f4080ecc3df3d6430bf6f8ec8abca00c90c8b990ffbfdd72dc2156d'
SUMMARY_SHA='4e9b32fb7fc08cbd5f8b97f210ed1dfc275d58811b91260409c41520c5672737'; FULL_SHA='94aa48b5e48ea6e3bc47b0a81b4c9ea830ae8355eede551f56366e1e82c33d1b'
OLD_PROTECTION=H/'all-quality-1009-cap-interior-fit-022/preflight-003/report.json'; OLD_PROTECTION_SHA='8b36b1bbae96a27f4a358f5bf705651a0662b3145f3f64ae676703b389cc3a56'
OWNER=H/'all-quality-1009-banks-point-update-001/emit_and_replay_point_updates_001.py'; OWNER_SHA='c75a3b5532700df673113ce6ae7b069042244569f4abf03c714ad0953c5098b7'
RUN=H/'joint-local-i-patch-preflight-002/run-005-root-inventory'; QG=1e-5; MOVE_CAP=250e-6
CUT_GAP={0:0.02907209025018013,3:0.00476202578177310}; RHO={0:900.,3:900.}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def npz(p,keys=None):
 with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in (z.files if keys is None else keys)}
def imp_owner():
 if sha(OWNER)!=OWNER_SHA: raise ValueError('point-update owner source pin mismatch')
 s=importlib.util.spec_from_file_location('point_update_owner_1009',OWNER);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def scalar(a):
 x=np.asarray(a)
 if x.size!=1: raise ValueError('scalar pin shape')
 return str(x.reshape(-1)[0])
def component_rows(item):
 p=Path(item['target_rows_path'])
 if sha(p)!=item['target_rows_sha256']: raise ValueError('component target CSV hash mismatch')
 with p.open(newline='') as f:
  rd=csv.DictReader(f)
  if rd.fieldnames!=['child_row']: raise ValueError('unexpected component CSV schema')
  a=np.fromiter((int(x['child_row']) for x in rd),np.int64)
 if len(a)!=int(item['target_child_count']) or len(np.unique(a))!=len(a): raise ValueError('component CSV count/unique failure')
 return a
def rim_vertices(faces):
 f=np.asarray(faces,np.int64);e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));e.sort(axis=1);u,c=np.unique(e,axis=0,return_counts=True);return set(map(int,np.unique(u[c==1])))
def protection(owner,curi):
 if sha(OLD_PROTECTION)!=OLD_PROTECTION_SHA: raise ValueError('protected-scope report pin mismatch')
 d=json.loads(OLD_PROTECTION.read_text())
 for k in ('back_scope','route_inventory'):
  x=d[k]
  if sha(Path(x['path']))!=x['sha256']: raise ValueError('protected mask pin mismatch '+k)
 oldpath=Path(d['input_state']['path'])
 if sha(oldpath)!=d['input_state']['sha256']: raise ValueError('protection parent state pin mismatch')
 _,s=owner.unwrap_state(json.loads(oldpath.read_text())); oldassets=owner.current_asset_pins(s); oldi=npz(Path(oldassets['I']['path']))
 if not owner.same_array(oldi['faces'],curi['faces']): raise ValueError('protected I face topology is stale')
 for k in curi:
  if k.startswith('face_ancestry') or k.startswith('face_source_') or k=='quality_star_face_output_to_input_row':
   if k not in oldi or not owner.same_array(oldi[k],curi[k]): raise ValueError('protected I ancestry mismatch '+k)
 with np.load(d['back_scope']['path'],allow_pickle=False) as z: back=set(map(int,z['I_vertex_rows']))
 rd=json.loads(Path(d['route_inventory']['path']).read_text()); pts=np.asarray(curi['points_world_m_f32'],np.float32); routes=set()
 for x in rd.get('surface_routes',[]):
  if x.get('role')!='I': continue
  if 'surface_vertex_row' in x: r=int(x['surface_vertex_row'])
  elif 'target_world_m_f32' in x:
   hit=np.flatnonzero(np.all(pts==np.asarray(x['target_world_m_f32'],np.float32),axis=1))
   if len(hit)!=1: raise ValueError('route coordinate is not a unique I row')
   r=int(hit[0])
  else: raise ValueError('route lacks I row key')
  routes.add(r)
 rims=rim_vertices(curi['faces'])
 for name,ss in [('back',back),('routes',routes),('rims',rims)]:
  if any(x<0 or x>=len(pts) for x in ss): raise ValueError(name+' row outside I points')
 return back,routes,rims

def grad_for(p,t,rows,n):
 tt=np.asarray(t[rows],np.int64);x=np.asarray(p[tt],np.float64);a,b,c,d=x[:,0],x[:,1],x[:,2],x[:,3]
 gb=np.cross(c-a,d-a)/6.;gc=np.cross(d-a,b-a)/6.;gd=np.cross(b-a,c-a)/6.;ga=-(gb+gc+gd)
 g=np.zeros((n,3),np.float64);np.add.at(g,tt.reshape(-1),np.stack((ga,gb,gc,gd),axis=1).reshape(-1,3))
 det=np.einsum('ij,ij->i',b-a,np.cross(c-a,d-a));return g,float(det.sum(dtype=np.float64)/6.),det

def smooth(field,faces,protected,passes=2):
 x=np.asarray(field,np.float64).copy();f=np.asarray(faces,np.int64);e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));e.sort(axis=1);e=np.unique(e,axis=0);a,b=e[:,0],e[:,1];deg=np.bincount(np.r_[a,b],minlength=len(x)).astype(np.float64)
 for _ in range(passes):
  acc=np.zeros_like(x);np.add.at(acc,a,x[b]);np.add.at(acc,b,x[a]);x=.5*x+.5*acc/np.maximum(deg[:,None],1.)
  if protected:x[np.fromiter(protected,np.int64)]=0.
 return x

def volume_sum(p,t,rows):
 x=np.asarray(p[t[rows]],np.float64);return float(np.sum(np.einsum('ij,ij->i',x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))/6.,dtype=np.float64))
def face_checks(old,new,faces,rows):
 f=np.asarray(faces,np.int64);hit=np.flatnonzero(np.isin(f,np.asarray(sorted(rows),np.int64)).any(axis=1));x=np.asarray(old[f[hit]],np.float64);y=np.asarray(new[f[hit]],np.float64);on=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);nn=np.cross(y[:,1]-y[:,0],y[:,2]-y[:,0]);oa=np.linalg.norm(on,axis=1);na=np.linalg.norm(nn,axis=1);co=np.einsum('ij,ij->i',on,nn)/np.maximum(oa*na,np.finfo(float).tiny);ar=na/np.maximum(oa,np.finfo(float).tiny);rs=[{'row':int(r),'orientation_cosine':float(c),'area_ratio':float(q)} for r,c,q in zip(hit,co,ar)];return rs,{'count':len(hit),'min_cosine':float(co.min(initial=1.)),'min_area_ratio':float(ar.min(initial=1.)),'max_area_ratio':float(ar.max(initial=1.)),'pass':bool(np.all(co>=.5) and np.all(ar>=.5) and np.all(ar<=2.))}
def exact_star(owner,a,b,t,cells):
 oldbad=[];newbad=[]
 for c in cells:
  if owner.exact_det6(a,t[int(c)])<=0:oldbad.append(int(c))
  if owner.exact_det6(b,t[int(c)])<=0:newbad.append(int(c))
 return oldbad,newbad

def main():
 if RUN.exists(): raise FileExistsError(RUN)
 for p,h in [(STATE,STATE_SHA),(TYPED,TYPED_SHA),(FULL,FULL_SHA),(SUMMARY,SUMMARY_SHA),(OWNER,OWNER_SHA)]:
  if sha(p)!=h: raise ValueError('frozen input pin mismatch '+str(p))
 owner=imp_owner();raw=json.loads(STATE.read_text());_,state=owner.unwrap_state(raw)
 if state.get('status')!='PASS_ZERO_SKIN_FAT_SOURCE_QUALITY_MASS_PENDING':raise ValueError('selected 1020 state status mismatch')
 assets=owner.current_asset_pins(state)
 for role,path in {'I':I_PATH,'O':O_PATH,'D':D_PATH,'Skin':SKIN_PATH,'Fat':FAT_PATH}.items():
  if Path(assets[role]['path'])!=path or assets[role]['sha256']!=PINS[role] or sha(path)!=PINS[role]:raise ValueError('state asset mismatch '+role)
 c= npz(TYPED,['points_reference_world_m_f64','volume_tetrahedra','material_index','geometric_donor_body','anatomical_region_id','current_typed_role_active_source_vertex_rows_I','current_typed_role_source_row_to_joint_node_I','current_typed_role_active_source_vertex_rows_O','current_typed_role_source_row_to_joint_node_O','current_typed_role_active_source_vertex_rows_D','current_typed_role_source_row_to_joint_node_D','current_typed_child_label_reference_source_sha256','stable_source_face_id'])
 p64=np.asarray(c['points_reference_world_m_f64'],np.float64);p32=p64.astype(np.float32);t=np.asarray(c['volume_tetrahedra'],np.int64);mat=np.asarray(c['material_index'],np.int64);don=np.asarray(c['geometric_donor_body'],np.int64);reg=np.asarray(c['anatomical_region_id'],np.int64)
 if p64.shape!=(311768,3) or t.shape!=(1495294,4) or mat.shape!=(len(t),):raise ValueError('current 433 shape mismatch')
 if sha(LABELS)!=LABELS_SHA:raise ValueError('1044 child-label CSV hash mismatch')
 typed_label_ref_scalar=scalar(c['current_typed_child_label_reference_source_sha256'])
 label_keys=('material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id')
 label_csv={k:np.empty(len(t),np.int64) for k in label_keys}
 with LABELS.open(newline='') as f:
  rd=csv.DictReader(f)
  if rd.fieldnames!=['child_tet_row','material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id']:raise ValueError('unexpected 1044 labels CSV header')
  nlabel=0
  for ri,row in enumerate(rd):
   if int(row['child_tet_row'])!=ri:raise ValueError('1044 labels row identity/order mismatch')
   for k in label_keys:label_csv[k][ri]=int(row[k])
   nlabel=ri+1
 if nlabel!=len(t):raise ValueError('1044 labels row count mismatch')
 for k in label_keys:
  if not np.array_equal(label_csv[k],np.asarray(c[k],np.int64)):raise ValueError('current 433 label row parity failure: '+k)
 census=json.loads(CENSUS.read_text());summary=json.loads(SUMMARY.read_text())
 if census.get('status')!='PASS_SOURCE_ONLY_COMPONENT_CENSUS' or len(census['components'])!=11 or census.get('target_child_count')!=588632:raise ValueError('current027 census status/count mismatch')
 if summary.get('current027_manifest_sha256')!='c6ebc27b9944f8a39da4420f4a9c56fff6a0c70b34a6384322d8939bf9537b0e' or summary.get('typed_support_sha256')!='76479632eae7c84f6a905f741f7ec7e84dab0eba9f8ef94fe4336cef8d77f854':raise ValueError('027 provenance mismatch')
 comps=[]
 for x in census['components']:
  r=component_rows(x);m=int(mat[r[0]])
  if r.size and (r.min()<0 or r.max()>=len(t)):raise ValueError('target row outside current 433 domain')
  if not np.all(mat[r]==m):raise ValueError('027 target component mixed material; decompose by material before fitting')
  comps.append(r)
 full=np.loadtxt(FULL,skiprows=1,dtype=np.int64).reshape(-1);un=np.concatenate(comps)
 if len(np.unique(un))!=len(un) or not np.array_equal(np.sort(un),np.sort(full)):raise ValueError('11 target sets do not partition current027 full target')
 curi=npz(I_PATH);cur_o=npz(O_PATH);cur_d=npz(D_PATH);faces=np.asarray(curi['faces'],np.int64);ip=np.asarray(curi['points_world_m_f32'],np.float32)
 ai=np.asarray(c['current_typed_role_active_source_vertex_rows_I'],np.int64);ij=np.asarray(c['current_typed_role_source_row_to_joint_node_I'],np.int64)
 if not np.array_equal(np.sort(ai),np.unique(faces.reshape(-1))):raise ValueError('active I source rows != face domain')
 if not np.array_equal(p32[ij[ai]].view(np.uint32),ip[ai].view(np.uint32)):raise ValueError('I source→433 exact F32 coordinate map fails')
 parents={'Skin':npz(SKIN_PATH),'Fat':npz(FAT_PATH)};source_maps=owner.check_source_maps(curi,parents)
 back,routes,rims=protection(owner,curi);ao=np.asarray(c['current_typed_role_active_source_vertex_rows_O'],np.int64);oj=np.asarray(c['current_typed_role_source_row_to_joint_node_O'],np.int64);ad=np.asarray(c['current_typed_role_active_source_vertex_rows_D'],np.int64);dj=np.asarray(c['current_typed_role_source_row_to_joint_node_D'],np.int64)
 od=set(map(int,np.r_[oj[ao],dj[ad]]));alias={int(r) for r in ai if int(ij[r]) in od};prot=back|routes|rims|alias
 qbase=owner.native_q(p32,t)
 if np.any(~np.isfinite(qbase)) or np.any(qbase<owner.Q_GATE):raise ValueError('current 433 baseline q gate is not clean')
 gi=np.zeros((11,len(ai),3),np.float64);vol=[];touched=[];stats=[]
 for cid,rows in enumerate(comps):
  g,v,ds=grad_for(p64,t,rows,len(p64))
  if np.any(ds<=0):raise ValueError('component has nonpositive F64 tet')
  gi[cid]=g[ij[ai]];nodes=np.unique(t[rows].reshape(-1));hit=np.flatnonzero(np.isin(ij[ai],nodes));touched.append(hit);vol.append(v)
  stats.append({'component_id':cid,'cell_count':len(rows),'node_count':len(nodes),'active_I_rows_touched':len(hit),'gradient_nonzero_active_I_rows':int(np.count_nonzero(np.linalg.norm(gi[cid],axis=1)>0))})
 RUN.mkdir(parents=True)
 arrays={'active_I_source_rows':ai,'active_I_joint_nodes':ij[ai],'gradients':gi,'volumes_m3':np.asarray(vol),'protected_source_I_rows':np.asarray(sorted(prot),np.int64)}
 for j,r in enumerate(comps):arrays['component_%02d_rows'%j]=r
 np.savez_compressed(RUN/'sensitivities.npz',**arrays)
 movable=np.asarray([int(r) not in prot for r in ai],bool)
 rows=[]
 for j,r in enumerate(comps):
  norm=np.linalg.norm(gi[j,movable],axis=1)
  rows.append({'component':j,'material':int(mat[r[0]]),'cells':len(r),'volume_m3':vol[j],'gradient_area_sum_m2':float(norm.sum()),'max_250um_volume_reduction_m3':float(250e-6*norm.sum()),'gradient_nonzero_rows':int(np.count_nonzero(norm))})
 result={'schema':'numi.h706.current1020.direct027-cut-volume-sensitivity-and-bounded-I-step.v1','status':'PASS_GRADIENT_INVENTORY_ONLY','physical_admission':False,'native_steps':0,'state':pin(STATE),'typed':pin(TYPED),'census':pin(CENSUS),'summary':pin(SUMMARY),'source':pin(Path(__file__)),'components':rows,'active_I_count':len(ai),'movable_I_count':int(movable.sum()),'protected_I_count':len(prot),'sensitivities':pin(RUN/'sensitivities.npz')}
 (RUN/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
