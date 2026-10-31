#!/usr/bin/env python3
"""Source-only paired reduction of the exact inferred D K4/Fat pocket."""
import hashlib, json, os, tempfile
from fractions import Fraction
from pathlib import Path
from collections import Counter
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'paired-inferred-envelope-resolution-reduction-293-root'
D_PATH=H/'native-apex-D-ears-282-root/run-001/typed-001/candidate-D-typed-current.npz'
D_RAW=H/'native-apex-D-ears-282-root/run-001/candidate-D-native-apex-connected-raw.npz'
D_REPORT=H/'native-apex-D-ears-282-root/run-001/report.json'
D015=H/'current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz'
FAT_PATH=H/'fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz'
FAT_REPORT=H/'fat-positive-PLC-prefix-compaction-135-root/report.json'
SUP=H/'fat-current-semantic-support-141-root/fat-current-semantic-support.npz'
SUP_REPORT=H/'fat-current-semantic-support-141-root/receipt.json'
SHELL=H/'fat-inferred-boundary-shell-assembly-140-v2/fat-shell-refined-typed.npz'
COLLAR=H/'microedge-collar-014-saved-op-replay/receipt.json'
PINS={
 D_PATH:'a8845bc3ab5106fa08e77ecd37f7bc65ffe9991961f68351c444e2cf6b1a6866',
 D_RAW:'4cb9aab4e303edac9f3f2e5fdfab7147cf3d78666d6f10b4c175138d2938e212',
 D015:'15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f',
 FAT_PATH:'7ab13a433e0cb08350c901388c0c86712269e0cf66ce70df5fe33de248357e76',
 SUP:'f3081e59fd88a88b48aa3d9caa8153609f6c24fcc0aac32d069c634ddad12b2b',
 SHELL:'65a13549560123b31ecbc698680edaff48ab843e834ef127e72f59d6aaffc04a',
}
D_FR=np.array([62075,62077,162298,162299],dtype=np.int64)
D_PR=np.array([56714,56715,56716,56717],dtype=np.int64)
FC=207166
FP=np.array([182082,182083,182084,182085],dtype=np.int64)
SF=np.array([316688,316690,418135,418136],dtype=np.int64)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def fr(x):
 n,d=float(x).as_integer_ratio();return Fraction(n,d)
def det3(a):
 return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def surfvol(P,F):
 z=Fraction(0)
 for tri in F:z+=det3([[fr(x) for x in row] for row in P[tri]])
 return z/6
def tetvol(P,t):
 p=P[t];return det3([[fr(p[i+1,j])-fr(p[0,j]) for j in range(3)] for i in range(3)])/6
def csrfilter(off,lanes,keep):
 oo=[0];ll=[[] for _ in lanes]
 for r in np.flatnonzero(keep):
  a,b=map(int,off[r:r+2])
  for out,x in zip(ll,lanes):out.extend(x[a:b].tolist())
  oo.append(len(ll[0]))
 return np.asarray(oo,dtype=off.dtype),[np.asarray(x,dtype=y.dtype) for x,y in zip(ll,lanes)]
def nested_filter(ro,keep,gids,po,prs,sn=None,sd=None,so=None,sr=None):
 ro2=[0];po2=[0];go=[];pr=[];sn2=[];sd2=[];so2=[0];sr2=[]
 for r in np.flatnonzero(keep):
  for j in range(int(ro[r]),int(ro[r+1])):
   go.append(int(gids[j]))
   if sn is not None:sn2.append(int(sn[j]));sd2.append(int(sd[j]))
   a,b=map(int,po[j:j+2]);pr.extend(prs[a:b].tolist());po2.append(len(pr))
   if so is not None:
    a,b=map(int,so[j:j+2]);sr2.extend(sr[a:b].tolist());so2.append(len(sr2))
  ro2.append(len(go))
 out={'ro':np.asarray(ro2,dtype=ro.dtype),'gids':np.asarray(go,dtype=gids.dtype),'po':np.asarray(po2,dtype=po.dtype),'pr':np.asarray(pr,dtype=prs.dtype)}
 if sn is not None:out.update(sn=np.asarray(sn2,dtype=sn.dtype),sd=np.asarray(sd2,dtype=sd.dtype))
 if so is not None:out.update(so=np.asarray(so2,dtype=so.dtype),sr=np.asarray(sr2,dtype=sr.dtype))
 return out
def save(path,arrays):
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd);tmp=Path(tmp);tmp.unlink()
 try:
  np.savez_compressed(tmp,**arrays); actual=tmp if tmp.exists() else Path(str(tmp)+'.npz');os.replace(actual,path)
 finally:
  for q in (tmp,Path(str(tmp)+'.npz')):
   if q.exists():q.unlink()
def closure(F):
 e=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])).astype(np.int64,copy=False)
 lo=np.minimum(e[:,0],e[:,1]).astype(np.uint64);hi=np.maximum(e[:,0],e[:,1]).astype(np.uint64)
 k=(lo<<np.uint64(20))|hi;s=np.where(e[:,0]<e[:,1],1,-1).astype(np.int8)
 u,inv,c=np.unique(k,return_inverse=True,return_counts=True);w=np.bincount(inv,weights=s,minlength=len(u))
 return {'unique_edges':int(len(u)),'edge_incidence_not_two':int(np.count_nonzero(c!=2)),'oriented_edge_sum_not_zero':int(np.count_nonzero(w!=0))}
def directed_edge_census(F):
 out={}
 for t in F:
  for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):
   e=tuple(sorted((a,b)));c,s=out.get(e,(0,0));out[e]=(c+1,s+(1 if a<b else -1))
 return out
def inherited_rim_wall_pairing(points,faces,open_edges,wallmesh):
 coord={}
 for i,p in enumerate(wallmesh['points_world_m']):
  coord.setdefault(tuple(np.asarray(p).tolist()),[]).append(i)
 used={int(x) for edge,_,_ in open_edges for x in edge}
 pmap={}
 unmatched=[]
 for i in used:
  hits=coord.get(tuple(np.asarray(points[i]).tolist()),[])
  if len(hits)==1:pmap[i]=hits[0]
  else:unmatched.append({'D_point_row':i,'exact_shell_rows':hits})
 if unmatched:raise RuntimeError(f'{len(unmatched)} used D point rows lack unique exact shell140 coordinate matches')
 wallrows=np.flatnonzero(wallmesh['face_source_role']=='donor_fat_ID_headwall')
 wed={}
 for fi in wallrows:
  t=wallmesh['faces'][fi]
  for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):
   e=tuple(sorted((a,b)));wed.setdefault(e,[]).append((int(fi),a,b))
 records=[];orientation={'same':0,'opposite':0,'other':0}
 for edge,c,s in open_edges:
  u,v=edge
  candidates=wed.get(tuple(sorted((pmap[u],pmap[v]))),[])
  if len(candidates)!=1:raise RuntimeError(f'D open edge {edge} has {len(candidates)} registered Fat rim-wall edge matches')
  fi,a,b=candidates[0]
  if (a,b)==(pmap[u],pmap[v]):rel='same'
  elif (a,b)==(pmap[v],pmap[u]):rel='opposite'
  else:rel='other'
  orientation[rel]+=1
  records.append({'D_point_edge':[int(u),int(v)],'D_open_edge_incidence':int(c),'D_open_edge_signed_sum':int(s),'shell140_wall_face_row':fi,'shell140_wall_directed_edge':[a,b],'orientation_relation':rel})
 return {'owner':'fat-inferred-boundary-shell-assembly-140-v2 donor_fat_ID_headwall faces, cross-bound to microedge-collar-014 wall owner receipt','wall_face_rows':int(len(wallrows)),'D_open_edge_count':len(open_edges),'exact_unique_coordinate_mapping_for_used_D_points':True,'every_D_open_edge_has_exactly_one_wall_edge':True,'orientation_relation_counts':orientation,'oriented_edge_cancellation_established':bool(orientation['opposite']==len(open_edges)),'raw_D_vs_Fat_inner_rim_orientation_note':'Raw same-direction pairing is recorded without claiming physical cancellation; the Fat inner-D role may reverse D at assembly, and the existing combined owner must verify that.' ,'records':records}
def main():
 for p,expected in PINS.items():
  got=sha(p)
  if got!=expected:raise RuntimeError(f'input pin mismatch {p}: {got}')
 OUT.mkdir(parents=True,exist_ok=True)
 outputs=[OUT/x for x in ['candidate-D-paired-resolution-reduced.npz','candidate-Fat-paired-resolution-reduced.npz','fat-semantic-support-paired-resolution-reduced.npz','paired-retirement-maps.npz','operation.json','receipt.json']]
 if any(x.exists() for x in outputs):raise RuntimeError('refuse to overwrite existing output')
 d=load(D_PATH);d0=load(D015);v=load(FAT_PATH);s=load(SUP);sh=load(SHELL)
 drep=json.loads(D_REPORT.read_text());vrep=json.loads(FAT_REPORT.read_text());srep=json.loads(SUP_REPORT.read_text());collar=json.loads(COLLAR.read_text())
 df=d['faces'];dp=d['points_world_m'];nf,npnt=len(df),len(dp)
 if not np.array_equal(np.unique(df[D_FR]),D_PR):raise RuntimeError('D K4 point rows mismatch')
 other=np.setdiff1d(np.flatnonzero(np.any(np.isin(df,D_PR),axis=1)),D_FR)
 if len(other):raise RuntimeError('D K4 vertices used by other D faces')
 edge=[]
 for t in df[D_FR]:
  edge.extend(tuple(sorted((int(a),int(b)))) for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])])
 ec=Counter(edge)
 if len(ec)!=6 or set(ec.values())!={2}:raise RuntimeError('D K4 is not a closed 4-face shell')
 dv=surfvol(dp,df[D_FR])
 if dv>=0:raise RuntimeError('D K4 does not have negative winding')
 dsource=d['quality_star_face_output_to_input_row'][D_FR].astype(np.int64)
 fields=['face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1']
 qfields=['quality_star_patch_union_face_source_layer','quality_star_patch_union_face_source_body','quality_star_patch_union_face_source_bone','quality_star_patch_union_face_source_face_id','quality_star_patch_union_face_source_original_id','quality_star_patch_union_face_source_identity0','quality_star_patch_union_face_source_identity1']
 schema=json.loads(str(d0['face_source_identity_schema_by_original_id_json'][0]))
 source_schema={x:schema[x] for x in ['100000060','100000065']}
 for x,o in source_schema.items():
  if o.get('role')!='BONE' or o.get('envelope_operation')!='Minkowski sum with centered 4 um cube' or float(o.get('envelope_radius_m',0))!=2e-6:raise RuntimeError('D inference registry mismatch')
 union=set();drecs=[]
 for r,sr in zip(D_FR,dsource):
  if int(d['face_ancestry_source_face_rows'][d['face_ancestry_offsets'][r]])!=int(sr):raise RuntimeError('D face source/ancestry mismatch')
  if sorted(map(tuple,dp[df[r]].tolist()))!=sorted(map(tuple,d0['points_world_m'][d0['faces'][sr]].tolist())):raise RuntimeError('D015 source geometry mismatch')
  typed=tuple(int(d[k][r]) for k in fields)
  if typed!=tuple(int(d0[k][sr]) for k in fields):raise RuntimeError('D015 typed tuple mismatch')
  a,b=map(int,d['quality_star_patch_union_offsets'][r:r+2])
  tuples=[tuple(int(d[k][j]) for k in qfields) for j in range(a,b)]
  if typed not in tuples:raise RuntimeError('D typed union incomplete')
  union.update(tuples)
  a,b=map(int,d['face_ancestry_offsets'][r:r+2])
  ancestry=[[int(d['face_ancestry_source_face_rows'][j]),int(d['face_ancestry_source_original_ids'][j]),int(d['face_ancestry_parent_face_rows'][j])] for j in range(a,b)]
  drecs.append({'D282_face_row':int(r),'D015_source_face_row':int(sr),'typed_tuple':list(typed),'full_typed_union':tuples,'source_ancestry':ancestry,'vertices':df[r].astype(int).tolist()})
 protected={int(x['surface_vertex_row']) for x in collar['protected149']['surface_routes'] if x.get('role')=='D'}
 if set(D_PR.tolist())&protected:raise RuntimeError('K4 overlaps protected149')
 vpts=v['points_world_m'];tets=v['tetrahedra'];mats=v['material_indices'];tet=tets[FC].astype(np.int64)
 if set(tet.tolist())!=set(FP.tolist()):raise RuntimeError('Fat counterpart point rows mismatch')
 coordkey=lambda a:sorted(tuple(float(c) for c in x) for x in a)
 if coordkey(vpts[tet])!=coordkey(dp[D_PR]):raise RuntimeError('D/Fat pocket coordinates differ')
 fv=tetvol(vpts,tet)
 if fv<=0 or fv!=-dv:raise RuntimeError('Fat tet not exact positive opposite of D K4')
 if int(mats[FC])!=19:raise RuntimeError('raw Fat attribute changed')
 a,b=map(int,s['cell_group_offsets'][FC:FC+2])
 if b-a!=1:raise RuntimeError('Fat tet has nonunique semantic group')
 gid=int(s['cell_group_ids'][a]);semantic=tuple(int(x) for x in s['group_tuples'][gid])
 if semantic!=(16,23,23):raise RuntimeError(f'Fat semantic support mismatch {semantic}')
 if int(s['cell_group_p1_score_numerators'][a])!=4 or int(s['cell_group_p1_score_denominators'][a])!=4:raise RuntimeError('Fat P1 support not full')
 node_usage={int(n):np.flatnonzero(np.any(tets==n,axis=1)).astype(int).tolist() for n in tet}
 if any(x!=[FC] for x in node_usage.values()):raise RuntimeError('Fat counterpart not isolated')
 shellrows=sh['faces']; shellsort=np.sort(shellrows,axis=1)
 if not np.array_equal(vpts[tet],sh['points_world_m'][tet]):raise RuntimeError('Fat shell point coordinates mismatch')
 tetfaces=[(tet[1],tet[2],tet[3]),(tet[0],tet[3],tet[2]),(tet[0],tet[1],tet[3]),(tet[0],tet[2],tet[1])]
 match=[]
 for tri in tetfaces:
  q=np.sort(np.asarray(tri,dtype=shellrows.dtype));hits=np.flatnonzero(np.all(shellsort==q,axis=1))
  if len(hits)!=1:raise RuntimeError(f'Fat counterpart boundary shell matches={len(hits)}')
  match.append(int(hits[0]))
 if set(match)!=set(SF.tolist()):raise RuntimeError(f'Fat shell rows mismatch: {match}')
 if np.count_nonzero(np.any(np.isin(shellrows,FP),axis=1))!=4:raise RuntimeError('Fat shell points touch outside counterpart')
 sfrec=[]
 for r in match:
  if str(sh['face_source_role'][r])!='inner_D':raise RuntimeError('Fat shell face not inner_D')
  if int(sh['face_source_original_id'][r]) not in [100000060,100000065]:raise RuntimeError('Fat shell source ID mismatch')
  a,b=map(int,sh['face_ancestry_offsets'][r:r+2])
  ans=[[int(sh['face_ancestry_source_rows'][j]),int(sh['face_ancestry_source_original_ids'][j]),int(sh['face_ancestry_parent_face_rows'][j])] for j in range(a,b)]
  sfrec.append({'Fat140_shell_face_row':r,'vertices':shellrows[r].astype(int).tolist(),'role':'inner_D','source_original_id':int(sh['face_source_original_id'][r]),'source_row':int(sh['face_source_row'][r]),'source_face_id':int(sh['face_source_face_id'][r]),'ancestry':ans})
 if surfvol(sh['points_world_m'],shellrows[match])!=fv:raise RuntimeError('Fat shell interface orientation/volume mismatch')
 if s['volume_sha256'].item()!=sha(FAT_PATH):raise RuntimeError('Fat semantic support volume pin mismatch')
 source330=srep['source330'];masspath=Path(source330['massPath'])
 if sha(masspath)!=source330['massSHA256']:raise RuntimeError('source330 mass table pin mismatch')
 if source330['massSHA256']!='c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679':raise RuntimeError('unexpected source330 mass table')
 # Filter D points/faces and all aligned typed CSRs.
 kp=np.ones(npnt,dtype=bool);kp[D_PR]=False
 pmap=np.full(npnt,-1,dtype=np.int64);pmap[kp]=np.arange(kp.sum())
 kf=np.ones(nf,dtype=bool);kf[D_FR]=False
 fmap=np.full(nf,-1,dtype=np.int64);fmap[kf]=np.arange(kf.sum())
 qoff,ql=csrfilter(d['quality_star_patch_union_offsets'],[d[k] for k in qfields],kf)
 afields=['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows']
 aoff,al=csrfilter(d['face_ancestry_offsets'],[d[k] for k in afields],kf)
 pkeys={'points_world_m','points_world_m_f32','quality_star_point_output_to_input_row'}
 faligned=set(fields+['quality_star_face_output_to_input_row'])
 dout={}
 for k,x in d.items():
  if k in pkeys:dout[k]=x[kp]
  elif k=='faces':
   dout[k]=pmap[x[kf]]
   if np.any(dout[k]<0):raise RuntimeError('retained D face references removed point')
  elif k in faligned:dout[k]=x[kf]
  elif k in ['quality_star_patch_union_offsets','face_ancestry_offsets']+qfields+afields:continue
  else:dout[k]=x.copy()
 dout['quality_star_patch_union_offsets']=qoff
 dout.update(dict(zip(qfields,ql)));dout['face_ancestry_offsets']=aoff;dout.update(dict(zip(afields,al)))
 dclose_before=closure(df)
 dclose=closure(dout['faces'])
 old_edge_census=directed_edge_census(df)
 retained_parent_faces=df[kf]
 new_output_as_parent= np.flatnonzero(kp)[dout['faces']]
 retained_census=directed_edge_census(retained_parent_faces)
 output_census=directed_edge_census(new_output_as_parent)
 if retained_census != output_census:raise RuntimeError('retained D directed-edge census changed under point/face compaction')
 old_boundary={e:v for e,v in old_edge_census.items() if v[0]!=2 or v[1]!=0}
 new_boundary={e:v for e,v in output_census.items() if v[0]!=2 or v[1]!=0}
 if old_boundary != new_boundary:raise RuntimeError('D boundary edge incidence/direction witness changed under paired retirement')
 open_edges=[(e,c,s) for e,(c,s) in old_boundary.items() if c==1]
 if len(open_edges)!=45:raise RuntimeError(f'expected 45 inherited collar open edges, found {len(open_edges)}')
 rim_wall_pairing=inherited_rim_wall_pairing(dp,df,open_edges,sh)
 # Filter Fat cell/points, preserve ordered tets/materials and update parent maps.
 kp2=np.ones(len(vpts),dtype=bool);kp2[FP]=False
 pmap2=np.full(len(vpts),-1,dtype=np.int64);pmap2[kp2]=np.arange(kp2.sum())
 kc=np.ones(len(tets),dtype=bool);kc[FC]=False
 cmap=np.full(len(tets),-1,dtype=np.int64);cmap[kc]=np.arange(kc.sum())
 vout={k:x.copy() for k,x in v.items()}
 vout['points_world_m']=vpts[kp2];vout['tetrahedra']=pmap2[tets[kc]];vout['material_indices']=mats[kc]
 if np.any(vout['tetrahedra']<0):raise RuntimeError('retained Fat tet references removed point')
 if 'point_output_to_input_rows' in v:vout['point_output_to_input_rows']=v['point_output_to_input_rows'][kp2]
 if 'point_input_to_output_row' in v:
  old=v['point_input_to_output_row'].astype(np.int64);inv=np.full(old.shape,-1,dtype=np.int64);ok=(old>=0)&(old<len(pmap2));inv[ok]=pmap2[old[ok]];vout['point_input_to_output_row']=inv.astype(v['point_input_to_output_row'].dtype)
 if 'cell_output_to_input_rows' in v:vout['cell_output_to_input_rows']=v['cell_output_to_input_rows'][kc]
 for k in ['promoted_inferred_D_boundary_output_node_rows']:
  if k in v:
   q=pmap2[v[k].astype(np.int64)]
   if np.any(q<0):raise RuntimeError('retired point is protected promoted-D row')
   vout[k]=q.astype(v[k].dtype)
 orig=int(v['original_PLC_point_count'].reshape(-1)[0]);new=int(v['new_PLC_point_count'].reshape(-1)[0])
 ro=int(np.count_nonzero(FP<orig));rn=int(np.count_nonzero(FP<new))
 if ro!=4 or rn!=4:raise RuntimeError('retired Fat points are not all in PLC prefix')
 vout['original_PLC_point_count']=np.asarray([orig-ro],dtype=v['original_PLC_point_count'].dtype)
 vout['new_PLC_point_count']=np.asarray([new-rn],dtype=v['new_PLC_point_count'].dtype)
 # Reindex nested node and cell semantic-support CSRs.
 if len(s['node_group_offsets'])-1!=len(vpts) or len(s['cell_group_offsets'])-1!=len(tets):raise RuntimeError('support row domain mismatch')
 nc=nested_filter(s['node_group_offsets'],kp2,s['node_group_ids'],s['node_group_parent_tet_offsets'],s['node_group_parent_tet_rows'])
 cc=nested_filter(s['cell_group_offsets'],kc,s['cell_group_ids'],s['cell_group_parent_tet_offsets'],s['cell_group_parent_tet_rows'],s['cell_group_p1_score_numerators'],s['cell_group_p1_score_denominators'],s['cell_group_stable_face_offsets'],s['cell_group_stable_face_rows'])
 sout={k:x.copy() for k,x in s.items()}
 sout.update(node_group_offsets=nc['ro'],node_group_ids=nc['gids'],node_group_parent_tet_offsets=nc['po'],node_group_parent_tet_rows=nc['pr'])
 for k in ['node_origin_kind','node_parent_registration_distance_m','node_parent_chosen_tet_row','node_parent_barycentric','node_parent_residual_world_m']:sout[k]=s[k][kp2]
 sout.update(cell_group_offsets=cc['ro'],cell_group_ids=cc['gids'],cell_group_parent_tet_offsets=cc['po'],cell_group_parent_tet_rows=cc['pr'],cell_group_p1_score_numerators=cc['sn'],cell_group_p1_score_denominators=cc['sd'],cell_group_stable_face_offsets=cc['so'],cell_group_stable_face_rows=cc['sr'])
 DOUT=OUT/'candidate-D-paired-resolution-reduced.npz';VOUT=OUT/'candidate-Fat-paired-resolution-reduced.npz';SOUT=OUT/'fat-semantic-support-paired-resolution-reduced.npz';MOUT=OUT/'paired-retirement-maps.npz'
 save(DOUT,dout);save(VOUT,vout);sout['volume_sha256']=np.asarray(sha(VOUT),dtype='<U64');save(SOUT,sout)
 maps={
  'D_output_point_to_parent282_row':np.flatnonzero(kp).astype(np.int64),'D_parent282_point_to_output_row':pmap,
  'D_output_face_to_parent282_row':np.flatnonzero(kf).astype(np.int64),'D_parent282_face_to_output_row':fmap,
  'Fat_output_point_to_parent135_row':np.flatnonzero(kp2).astype(np.int64),'Fat_parent135_point_to_output_row':pmap2,
  'Fat_output_cell_to_parent135_row':np.flatnonzero(kc).astype(np.int64),'Fat_parent135_cell_to_output_row':cmap,
  'Fat_output_point_to_source127_row':vout.get('point_output_to_input_rows',np.full(kp2.sum(),-1,dtype=np.int64)).astype(np.int64),
  'Fat_output_cell_to_source127_row':vout.get('cell_output_to_input_rows',np.full(kc.sum(),-1,dtype=np.int64)).astype(np.int64),
 }
 save(MOUT,maps)
 # Readback hashes/row domains and source-only receipts.
 if len(load(DOUT)['faces'])!=nf-4 or len(load(DOUT)['points_world_m'])!=npnt-4:raise RuntimeError('D output counts wrong')
 vs=load(VOUT);ss=load(SOUT)
 if len(vs['tetrahedra'])!=len(tets)-1 or len(vs['points_world_m'])!=len(vpts)-4:raise RuntimeError('Fat output counts wrong')
 if ss['volume_sha256'].item()!=sha(VOUT):raise RuntimeError('support volume hash not rebound')
 if len(ss['cell_group_offsets'])-1!=len(vs['tetrahedra']) or len(ss['node_group_offsets'])-1!=len(vs['points_world_m']):raise RuntimeError('support coverage wrong')
 group=[]
 for j in range(a,b):
  p0,p1=map(int,s['cell_group_parent_tet_offsets'][j:j+2]);q0,q1=map(int,s['cell_group_stable_face_offsets'][j:j+2])
  group.append({'group_id':int(s['cell_group_ids'][j]),'tuple':[int(x) for x in s['group_tuples'][int(s['cell_group_ids'][j])]],'P1_score':[int(s['cell_group_p1_score_numerators'][j]),int(s['cell_group_p1_score_denominators'][j])],'source330_parent_tet_rows':s['cell_group_parent_tet_rows'][p0:p1].astype(int).tolist(),'stable_face_rows':s['cell_group_stable_face_rows'][q0:q1].astype(int).tolist()})
 vfloat=float(fv); volstr=str(fv)
 op={
  'schema':'numi.paired-inferred-envelope-resolution-reduction.v1','operation_id':'retire-negative-inferred-D-K4-and-coincident-positive-Fat-pocket',
  'status':('PASS_SOURCE_ONLY_PAIRED_SUBRESOLUTION_COMPONENT_RETIREMENT' if dclose['edge_incidence_not_two']==0 and dclose['oriented_edge_sum_not_zero']==0 else 'PASS_SOURCE_ONLY_PAIRED_RETIREMENT_WITH_INHERITED_D_OPEN_EDGE_WITNESS'),'physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN',
  'D':{'input_path':str(D_PATH),'input_sha256':sha(D_PATH),'retired_face_rows':D_FR.astype(int).tolist(),'retired_point_rows':D_PR.astype(int).tolist(),'retired_faces':drecs,'retired_complete_typed_union_7lanes':[list(x) for x in sorted(union)],'exact_signed_volume_m3':str(dv),'source_schema':source_schema,'protected149_overlap':[],'output_path':str(DOUT),'output_sha256':sha(DOUT),'output_maps':str(MOUT)},
  'Fat':{'input_path':str(FAT_PATH),'input_sha256':sha(FAT_PATH),'retired_cell_row':FC,'retired_cell_nodes':tet.astype(int).tolist(),'retired_point_rows':FP.astype(int).tolist(),'raw_region_attribute':int(v['material_indices'][FC]),'semantic_material_donor_region':list(semantic),'exact_positive_volume_m3':volstr,'volume_m3':vfloat,'isolated_tet_component':True,'node_incidence':node_usage,'matching_shell140_face_rows':match,'matching_shell_face_records':sfrec,'P1_source_support':group,'output_path':str(VOUT),'output_sha256':sha(VOUT),'semantic_support_path':str(SOUT),'semantic_support_sha256':sha(SOUT),'maps':str(MOUT)},
  'effect':{'D_closed_components_retired':1,'Fat_closed_components_retired':1,'D_faces_added':0,'Fat_cells_added':0,'D_all_remaining_faces_and_points_bitwise_preserved':True,'Fat_all_remaining_point_coordinates_bitwise_preserved':True,'Fat_remaining_tet_geometry_bitwise_preserved_after_point_row_map':True,'D_remaining_edge_closure':dclose,'D_boundary_edge_census_unchanged':True,'inherited_registered_rim_wall_pairing':rim_wall_pairing,'no_new_crossings_possible_by_set_deletion':True,'coincident_interface_before_retirement':True,'clearance_to_that_interface_m':0.0,'this_is_not_a_positive_clearance_claim':True},
  'mass':{'source330_mass_path':source330['massPath'],'source330_mass_sha256':source330['massSHA256'],'source330_mass_edited':False,'regional_mass_redistribution_required':True,'mass_mapping':'NOT_RUN','geometric_capacity_at_reference_density_1000_kg_m3_only':vfloat*1000.0,'density_rebind':'PENDING'},
  'proof_inputs':{'D_report_path':str(D_REPORT),'D_report_sha256':sha(D_REPORT),'Fat135_report_path':str(FAT_REPORT),'Fat135_report_sha256':sha(FAT_REPORT),'support141_receipt_path':str(SUP_REPORT),'support141_receipt_sha256':sha(SUP_REPORT),'protected149_receipt_path':str(COLLAR),'protected149_receipt_sha256':sha(COLLAR),'Fat_shell140_path':str(SHELL),'Fat_shell140_sha256':sha(SHELL)},
  'counts':{'D_points':[int(npnt),int(len(dout['points_world_m']))],'D_faces':[int(nf),int(len(dout['faces']))],'Fat_points':[int(len(vpts)),int(len(vout['points_world_m']))],'Fat_cells':[int(len(tets)),int(len(vout['tetrahedra']))],'Fat_support_nodes':[int(len(vpts)),int(len(ss['node_group_offsets'])-1)],'Fat_support_cells':[int(len(tets)),int(len(ss['cell_group_offsets'])-1)]},
  'physical_limitations':['source330 regional mass totals must be redistributed over the retained Fat domain','current Fat shell must be reassembled from final O/I/D typed owners before physical installation','no physical or native qualification is claimed']}
 (OUT/'operation.json').write_text(json.dumps(op,sort_keys=True,indent=2)+'\n')
 rec={'schema':'numi.paired-inferred-envelope-resolution-reduction-receipt.v1','status':op['status'],'physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','input_sha256':{str(p):sha(p) for p in PINS},'operation_path':str(OUT/'operation.json'),'operation_sha256':sha(OUT/'operation.json'),'outputs':{'D_candidate_sha256':sha(DOUT),'Fat_candidate_sha256':sha(VOUT),'Fat_support_sha256':sha(SOUT),'maps_sha256':sha(MOUT)},'counts':op['counts'],'checks':{'exact_negative_D_K4':True,'source_Minkowski_provenance_pinned':True,'no_protected149_overlap':True,'positive_Fat_semantic16_counterpart_exact':True,'Fat_counterpart_isolated_one_tet_component':True,'inner_D_shell_faces_match_all_four':True,'source330_mass_input_verified_and_untouched':True,'all_retained_geometry_unchanged':True,'remaining_D_surface_edge_closed':(dclose['edge_incidence_not_two']==0 and dclose['oriented_edge_sum_not_zero']==0),'inherited_D_open_edges_unchanged':True,'all_45_open_edges_match_registered_headwall_edges':True,'combined_oriented_edge_cancellation_verified':rim_wall_pairing['oriented_edge_cancellation_established'],'no_new_faces_or_cells':True,'paired_components_retired':True},'next_required':'Reassemble current Fat shell from final O/I/D owners, run conservative mass/density rebind preserving source330 regional totals, then existing physical gates.'}
 (OUT/'receipt.json').write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'status':rec['status'],'dir':str(OUT),'operation_sha256':sha(OUT/'operation.json'),'receipt_sha256':sha(OUT/'receipt.json'),'D_sha256':sha(DOUT),'Fat_sha256':sha(VOUT),'support_sha256':sha(SOUT),'maps_sha256':sha(MOUT),'D_closure':dclose,'exact_volume_m3':volstr,'counts':op['counts']}))
if __name__=='__main__':main()

