#!/usr/bin/env python3
"""Coupled microscopic I-interface endpoint quotients; geometry proposal only.
Outer skin/back and D boundaries remain unchanged. Skin/Fat share one exact I
quotient and their changed material volumes must cancel on the Float32 lattice.
"""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time,collections
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
top=load('surface209I352',H/'transfer_surface_contractions_to_volume_209_root.py');m=top.m
reg=load('exact309I352',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
def sha(p):
 with open(p,'rb') as q:return hashlib.file_digest(q,'sha256').hexdigest()
IP=H/'joint-native-OI-ears-279-root/I-batch002-run001/typed-280-002/candidate-I-typed-current.npz'
paths={'Skin':H/'cached-fan-continuation-344-root/skin-run001/pass-003/kernel-fan-volume-proposal.npz',
 'Fat':H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz'}
pins={'Skin':'c94316d73e3539be5bd4fd998f518060f1a586a3466c00994ca2a99f9c5db13f',
 'Fat':'d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f'}
out=H/'paired-inner-interface-quotient-352-root/run001';out.mkdir(exist_ok=False,parents=True);start=time.monotonic()
with np.load(IP) as z:sp=z['points_world_m_f32'];sf=z['faces']
edge=np.concatenate([sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]]]);se=np.sort(edge,axis=1)
uniq,counts=np.unique(se,axis=0,return_counts=True);rim=set(map(int,uniq[counts==1].ravel()))
flat=sf.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(sp)))]
def face_rows(n):return np.unique(order[offset[n]:offset[n+1]]//3)
def neighbours(n):return set(map(int,sf[face_rows(n)].ravel()))-{int(n)}
length=np.linalg.norm(sp[uniq[:,0]].astype(float)-sp[uniq[:,1]],axis=1)
candidate=uniq[(counts==2)&(length<=1e-5)]
vol={}
for role,path in paths.items():
 assert sha(path)==pins[role]
 with np.load(path) as z:v={k:z[k] for k in z.files}
 p=v['points_world_m'];t=v['tetrahedra'];mat=v['material_indices'];bd=top.boundary(t)
 look={top.key(x):i for i,x in enumerate(p)};assert len(look)==len(p)
 d,q=m.measurements(p,t);assert np.all(d>0)
 mp=np.asarray([look.get(top.key(x),-1) for x in sp],np.int64)
 hits=np.bincount(t[~m.passes(d,q)].ravel(),minlength=len(p))
 vol[role]={'path':path,'arrays':v,'p':p,'t':t.copy(),'mat':mat,'base_boundary':bd,
  'boundary_keys':{tuple(sorted(map(int,f))) for f in bd},'look':look,
  'active':np.ones(len(t),bool),'map':np.arange(len(p)),'accepted':[],'I_map':mp,
  'bad_hits':hits,'bad_before':int(np.sum(~m.passes(d,q))),'used':set()}
prep=H/'fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz'
with np.load(prep) as z:pending=z['global_point_rows']
protectedfat=set(map(int,pending));protectedfat.update(range(253194,len(vol['Fat']['p'])))
def score(pair):
 a,b=pair
 return sum(int(v['bad_hits'][v['I_map'][a]])+int(v['bad_hits'][v['I_map'][b]]) for v in vol.values() if v['I_map'][a]>=0 and v['I_map'][b]>=0)
candidate=sorted((tuple(map(int,e)) for e in candidate),key=lambda e:(-score(e),float(np.linalg.norm(sp[e[0]].astype(float)-sp[e[1]].astype(float))),e))
ops=[];reject=collections.Counter();usedI=set()
for a,b in candidate:
 if time.monotonic()-start>45 or len(ops)>=256:break
 if score((a,b))==0:continue
 if a in rim or b in rim:reject['PROTECTED_HEAD_RIM']+=1;continue
 ra=face_rows(a);rb=face_rows(b);edgefaces=np.intersect1d(ra,rb)
 if len(edgefaces)!=2 or (neighbours(a)&neighbours(b))!=set(map(int,sf[edgefaces].ravel()))-{a,b}:
  reject['SOURCE_SURFACE_LINK']+=1;continue
 for keep,drop in ((a,b),(b,a)):
  fr=face_rows(drop);verts=set(map(int,sf[fr].ravel()))
  if verts&usedI:reject['DEPENDENT_SOURCE_PATCH']+=1;continue
  source_new=np.where(sf[fr]==drop,keep,sf[fr])
  ds=np.any(np.diff(np.sort(source_new,axis=1),axis=1)==0,axis=1);source_new=source_new[~ds]
  if len(source_new)==0:reject['EMPTY_SURFACE_STAR']+=1;continue
  oldn=np.cross(sp[sf[fr]][:,1].astype(float)-sp[sf[fr]][:,0],sp[sf[fr]][:,2].astype(float)-sp[sf[fr]][:,0])[~ds]
  newxyz=sp[source_new].astype(float);newn=np.cross(newxyz[:,1]-newxyz[:,0],newxyz[:,2]-newxyz[:,0])
  den=np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1)
  if np.any(den<=0) or np.any(np.einsum('ij,ij->i',oldn,newn)<.9*den):
   reject['SOURCE_NORMAL_CHANGE']+=1;continue
  op={'retained_input_point_row':keep,'collapsed_input_point_row':drop,'retired_parent_face_rows':fr.tolist()}
  trials={};why=None
  for role,v in vol.items():
   trial,reason=top.try_op(v,sp,sf,op)
   if trial is None:why=role+':'+reason;break
   vv=set(map(int,trial['old'].ravel()))
   if vv&v['used']:why=role+':DEPENDENT_VOLUME_CAVITY';break
   if role=='Fat' and vv&protectedfat:why='PENDING_FAT_BOUNDARY_CAVITY';break
   trials[role]=trial
  if why is not None:reject[why]+=1;continue
  gain=sum(r['bad_before']-r['bad_after'] for r in trials.values())
  if gain<=0:reject['NO_STRICT_COMBINED_QUALITY_DESCENT']+=1;continue
  delta=0
  for role,r in trials.items():
   p=vol[role]['p']
   delta+=sum(reg.exact_star_determinants(p,r['new'][~r['deleted']]))-sum(reg.exact_star_determinants(p,r['old']))
  if delta!=0:reject['COMBINED_MATERIAL_VOLUME_CHANGED']+=1;continue
  for role,r in trials.items():
   v=vol[role];v['used'].update(map(int,r['old'].ravel()));v['t'][r['rows']]=r['new'];v['active'][r['rows'][r['deleted']]]=False
   v['map'][r['retired_volume_node']]=r['kept_volume_node']
   v['accepted'].append({'operation':len(ops),'source_cell_rows':r['rows'].tolist(),**{k:r[k] for k in ['kept_volume_node','retired_volume_node','removed_cells','bad_before','bad_after']}})
  usedI.update(verts)
  ops.append({**op,'edge_length_m':float(np.linalg.norm(sp[keep].astype(float)-sp[drop].astype(float))),
   'combined_native_quality_gain':gain,'exact_combined_Skin_Fat_volume_delta':0})
  break
mpI=np.arange(len(sp));retire=set()
for o in ops:mpI[o['collapsed_input_point_row']]=o['retained_input_point_row'];retire.update(o['retired_parent_face_rows'])
sourcefaces=mpI[sf];degen=np.any(np.diff(np.sort(sourcefaces,axis=1),axis=1)==0,axis=1)
sourcefaces=sourcefaces[~degen];sourcefile=out/'candidate-I-coupled-quotient-raw.npz'
np.savez_compressed(sourcefile,points_world_m=sp,faces=sourcefaces,source_input_face_rows=np.flatnonzero(~degen),source_input_vertex_quotient=mpI)
roles=[]
for role,v in vol.items():
 tt=v['t'][v['active']];bd=top.boundary(tt);expected=v['map'][v['base_boundary']]
 dd=np.any(np.diff(np.sort(expected,axis=1),axis=1)==0,axis=1);expected=expected[~dd]
 assert np.array_equal(top.boundary_identity(bd),top.boundary_identity(expected))
 assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
 det,q=m.measurements(v['p'],tt);assert np.all(det>0)
 bad=int(sum(~m.passes(det,q)));assert bad==v['bad_before']-sum(x['bad_before']-x['bad_after'] for x in v['accepted'])
 filename=out/(role.lower()+'-coupled-I-quotient-volume-proposal.npz')
 data=v['arrays'].copy();data.update(tetrahedra=tt,material_indices=v['mat'][v['active']],boundary_faces=bd,
  retained_input_cell_rows=np.flatnonzero(v['active']),source_I_vertex_quotient=mpI,source_interface_collapse_rows=np.arange(len(ops)))
 # desired surfaces remain unbound until typed I/context audit is regenerated.
 np.savez_compressed(filename,**data)
 roles.append({'role':role,'input':str(v['path']),'input_sha256':sha(v['path']),'output':str(filename),'output_sha256':sha(filename),
  'bad_before':v['bad_before'],'bad_after':bad,'nonpositive':0,'point_coordinates_bit_identical':True,
  'boundary_exactly_matches_paired_I_quotient':True,'operations':v['accepted']})
r={'status':'PASS_PAIRED_I_QUOTIENT_PHYSICAL_PREFLIGHT' if ops else 'NO_PAIRED_I_QUOTIENT_PROGRESS',
 'physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN',
 'source_I':str(IP),'source_I_sha256':sha(IP),'candidate_I':str(sourcefile),'candidate_I_sha256':sha(sourcefile),
 'maximum_edge_length_m':1e-5,'combined_Skin_Fat_exact_F32_volume_unchanged':True,
 'outer_O_and_D_geometry_unchanged':True,'source_coordinates_unchanged':True,'back_source_geometry_unchanged':True,
 'roles':roles,'passing_operations':len(ops),'operations':ops,'rejections':dict(reject),'producer_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-start,
 'limitation':'raw source I quotient requires typed seven-field parent unions, exact complete-context collision/apex audit and unchanged rim checks before use; no runtime/native or mass admission'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('operations','roles')}),flush=True)
for r in roles:print(json.dumps({k:v for k,v in r.items() if k!='operations'}),flush=True)
