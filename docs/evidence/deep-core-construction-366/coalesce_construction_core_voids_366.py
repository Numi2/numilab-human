#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,time,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');R=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/typed-D-62000100/run-002'
OPEN=R/'candidate-f32-typed.npz';CLOSED=R/'D-closed-csg-only.npz';CLASS=P/'deep-core-void-classification-362/receipt.json';VERIFY=R/'verification-receipt.json';OUT=P/'deep-core-void-coalescence-366'
PROD=P/'derived-lineage-producer-260/export_authoring_registered_surface_sidecars_230.py'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sv(x,f):
 q=x.astype(np.float64)-x[np.unique(f)].astype(np.float64).mean(0);a=q[f];return float(np.sum(np.einsum('fa,fa->f',a[:,0],np.cross(a[:,1],a[:,2]))/6,dtype=np.longdouble))
def edges(f):
 e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);k=np.sort(e,1);_,cnt=np.unique(k,axis=0,return_counts=True)
 return {'boundary_edges':int((cnt==1).sum()),'nonmanifold_edges':int((cnt>2).sum())}
def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False);a=load(OPEN);c=load(CLOSED);cl=json.loads(CLASS.read_text());ver=json.loads(VERIFY.read_text());x=a['points_world_m'];f=a['faces'];cf=c['faces']
 if sha(OPEN)!='ff2916e1ac63c2edb70a7440a9625412e618686aefcba2ec8d1904a3fce33d83' or sha(CLOSED)!=cl['inputs'][str(CLOSED)]:raise ValueError('candidate pin changed')
 if not np.array_equal(x,c['points_world_m']) or not np.array_equal(f,cf[:len(f)]) or len(cf)-len(f)!=41:raise ValueError('closed snapshot is not exact open prefix plus41caps')
 if cl['original_D_components']!=1 or cl['original_D_genus']!=0 or cl['void_outside_I328_m3']!=0 or cl['negative_shell_volume_sum_m3']>8e-9:raise ValueError('construction domain evidence does not support bounded proposal')
 e=np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]);graph=coo_matrix((np.ones(len(e),np.uint8),(e[:,0],e[:,1])),shape=(len(x),len(x))).tocsr()
 n,labels=connected_components(graph,directed=False);fl=labels[cf[:,0]]
 if not np.array_equal(labels[cf[:,1]],fl) or not np.array_equal(labels[cf[:,2]],fl):raise ValueError('component labels inconsistent')
 # Float32 coordinates are dyadic; classify closed-shell signed volume exactly.
 unit_exp=int(np.frexp(x[x!=0])[1].min())-24
 scaled=np.ldexp(x.astype(np.float64),-unit_exp)
 if not np.array_equal(scaled,np.rint(scaled)):raise ValueError('nonintegral dyadic lattice')
 ix=np.rint(scaled).astype(np.int64).astype(object);six_units=[]
 for k in range(n):
  ff=cf[fl==k];q=ix[ff]-ix[ff[0,0]]
  cr=q[:,1,1]*q[:,2,2]-q[:,1,2]*q[:,2,1]
  det=q[:,0,0]*cr+q[:,0,1]*(q[:,1,2]*q[:,2,0]-q[:,1,0]*q[:,2,2])+q[:,0,2]*(q[:,1,0]*q[:,2,1]-q[:,1,1]*q[:,2,0])
  six_units.append(int(det.sum()))
 volumes=np.array([float(v)*2.0**(3*unit_exp)/6.0 for v in six_units]);positive=np.array([v>0 for v in six_units]);negative=np.array([v<0 for v in six_units]);zero=np.array([v==0 for v in six_units])
 if n!=278 or positive.sum()!=3:raise ValueError(f'unexpected exact component inventory {n}/{positive.sum()}/{negative.sum()}/{zero.sum()}')
 keep=positive[fl[:len(f)]];drop=np.flatnonzero(~keep);kept=np.flatnonzero(keep)
 # Retain the three positive shells; every external boundary face and vertex coordinate is unchanged.
 used=np.unique(cf[positive[fl]].reshape(-1));node_map=np.full(len(x),-1,np.int64);node_map[used]=np.arange(len(used));new={}
 parent_offsets=a['face_ancestry_offsets'];parent_take=np.concatenate([np.arange(parent_offsets[i],parent_offsets[i+1]) for i in kept]);removed_take=np.concatenate([np.arange(parent_offsets[i],parent_offsets[i+1]) for i in drop]);ancestry_keys=[k for k in a if k.startswith('face_ancestry_') and k!='face_ancestry_offsets']
 for k,v in a.items():
  if k in ancestry_keys:new[k]=v[parent_take]
  elif k=='face_ancestry_offsets':new[k]=np.r_[0,np.cumsum(np.diff(v)[kept])]
  elif k=='faces':new[k]=node_map[v[kept]]
  elif k=='points_world_m' or k in ['source_f64_vertex_rows','emitted_vertex_input_row']:new[k]=v[used]
  elif k=='vertex_input_old_to_emitted':
   out=np.full(v.shape,-1,np.int64);ok=v>=0;out[ok]=node_map[v[ok]];new[k]=out
  elif k=='cap_boundary_vertex_rows':new[k]=node_map[v]
  elif k.startswith('face_') and v.ndim and len(v)==len(f):new[k]=v[kept]
  else:new[k]=v.copy()
 if (new['cap_boundary_vertex_rows']<0).any() or not np.array_equal(new['points_world_m'][new['faces']],x[f[kept]]):raise ValueError('external geometry changed')
 capnew=node_map[cf[len(f):]]
 if (capnew<0).any():raise ValueError('coalescence removed construction cap point')
 closednew=np.concatenate([new['faces'],capnew]);ee=edges(closednew)
 if ee!={'boundary_edges':0,'nonmanifold_edges':0}:raise ValueError('closed positive shells lost manifold edges')
 opath=OUT/'candidate-f32-typed.npz';np.savez_compressed(opath,**new)
 np.savez_compressed(OUT/'D-closed-csg-only.npz',points_world_m=new['points_world_m'],faces=closednew)
 removed={'parent_open_face_rows':drop,'parent_component_rows':fl[drop],'component_signed_volume_m3':volumes,'removed_faces_in_parent_vertex_domain':f[drop],'removed_face_ancestry_offsets':np.r_[0,np.cumsum(np.diff(parent_offsets)[drop])]}
 for k in ancestry_keys:removed[k]=a[k][removed_take]
 for k,v in a.items():
  if k.startswith('face_source_') and v.ndim and len(v)==len(f):removed[k]=v[drop]
 np.savez_compressed(OUT/'covered-construction-cavity-source-ancestry.npz',**removed)
 sys.path.insert(0,str(PROD.parent));spec=importlib.util.spec_from_file_location('producer260',PROD);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);valid=mod.validate_candidate(opath,'D',require_f32=True);valid={k:v for k,v in valid.items() if k not in ['points','faces','faceArrays','sourceSchema','faceAncestry']}
 added=-float(volumes[negative].sum());oldvol=sv(x,cf);newvol=sv(new['points_world_m'],closednew)
 if abs((newvol-oldvol)-added)>1e-15:raise ValueError('volume accounting mismatch')
 report={'status':'PASS_PROVENANCE_PRESERVING_CONSTRUCTION_VOID_COALESCENCE_CANDIDATE','inputs':{str(p):sha(p) for p in [OPEN,CLOSED,CLASS,VERIFY,PROD]},'candidate':str(opath),'candidate_sha256':sha(opath),'closed_csg_sha256':sha(OUT/'D-closed-csg-only.npz'),'covered_source_ancestry_sha256':sha(OUT/'covered-construction-cavity-source-ancestry.npz'),'old_closed_components':n,'retained_positive_components':int(positive.sum()),'covered_construction_voids':int(negative.sum()),'removed_zero_signed_volume_components':int(zero.sum()),'component_volume_sign_method':'exact integer determinant sum on Float32 dyadic lattice','dyadic_lattice_unit_exponent':unit_exp,'removed_open_faces':len(drop),'retained_open_faces':len(kept),'old_vertices':len(x),'retained_vertices':len(used),'added_excluded_core_volume_m3':added,'old_closed_volume_m3':oldvol,'new_closed_volume_m3':newvol,'maximum_vertex_displacement_m':0.0,'retained_face_geometry_bitwise_equal':True,'temporary_cap_faces':len(capnew),'closed_edge_audit':ee,'producer260_validation':valid,'exact_self_subset_certificate':{'parent_exact_self_pairs':ver['D_open']['full_exact_self_pair_count'],'retained_faces_are_exact_coordinate_subset':True,'new_triangles_created':0,'conclusion':'No new self intersections can arise by deleting faces from the verified parent with unchanged coordinates.'},'removed_void_volume_fraction_of_fat_reference':added/0.011289617863908516,'interpretation':'Explicit continuum-resolution construction repair for union-created microvoids, absent from the single genus0 original D core. Every removed boundary face remains in covered source ancestry. All positive external shells and construction caps retain coordinates exactly; no anatomy source file is edited. This candidate is not yet a volume mesh, mass transfer, contact, motion or physical admission certificate.','physical_admission':False,'physical_steps':0,'elapsed_s':time.monotonic()-t}
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
