#!/usr/bin/env python3
"""Retire inferred interior construction-node stars into an existing neighbour.
Final anatomical surface and all point coordinates/identities remain unchanged.
Only exact positive, strict native-quality-descent cavity replacements survive.
"""
from pathlib import Path
import argparse,importlib.util,json,hashlib,time,collections
import numpy as np
H=Path(__file__).resolve().parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('flip198collapse324',H/'fixed_boundary_volume_flip_198_root.py')
top=load('boundary209collapse324',H/'transfer_surface_contractions_to_volume_209_root.py')
reg=load('registered309collapse324',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def propose(p,t,node,target):
 old=np.asarray(t,np.int64)
 if node==target or target not in old:return None,'NOT_NEIGHBOUR'
 b=top.boundary(old)
 if np.any(b==node):return None,'NODE_ON_CAVITY_BOUNDARY'
 if not reg.star_boundary_closed(b):return None,'NONCLOSED_STAR'
 keep=~np.any(old==target,axis=1)
 new=old[keep].copy();new[new==node]=target
 if len(new)==0:return None,'EMPTY_REPLACEMENT'
 if len(np.unique(np.sort(new,axis=1),axis=0))!=len(new):return None,'DUPLICATE_REPLACEMENT'
 d,q=m.measurements(p,new)
 if np.any(d<=0):return None,'NONPOSITIVE_F32'
 oldbad=int(np.sum(~m.passes(*m.measurements(p,old))))
 newbad=int(np.sum(~m.passes(d,q)))
 if newbad>=oldbad:return None,'NO_STRICT_DEFECT_DESCENT'
 ok,why=m.certify(p,old,new)
 if not ok:return None,'EXACT_'+why
 # Sliding the interior point to target keeps every surviving determinant
 # positive and every removed edge-cell determinant positive until t=1.
 return (new,{'removed_interior_point_row':int(node),'target_point_row':int(target),
   'old_cells':len(old),'new_cells':len(new),'bad_before':oldbad,'bad_after':newbad,
   'minimum_normalized_det_after':float(q.min()),'minimum_volume_after_m3':float(d.min()/6),
   'exact_oriented_boundary_and_volume':True,'exact_F32_positive':True}),None
def fixture(out):
 out.mkdir(parents=True,exist_ok=False)
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[1e-9,1e-9,1e-9]],np.float32)
 outer=np.array([[0,1,2,3]],np.int64)
 t=m.orient(p,np.c_[np.full(4,4),top.boundary(outer)])
 ans,why=propose(p,t,4,0);assert ans is not None,why
 nt,op=ans;assert len(nt)==1 and op['bad_after']==0
 assert m.certify(p,t,nt)==(True,'PASS')
 ans,why=propose(p,t,0,4);assert ans is None and why=='NODE_ON_CAVITY_BOUNDARY'
 bad=p.copy();bad[4]=[.02,.02,.02]
 assert propose(bad,t,4,0)[0] is None
 r={'status':'PASS_INFERRED_INTERIOR_NODE_RETIREMENT_CONTROLS',
  'controls':['actual_4_to_1_cavity_path','exact_boundary_volume_positive','boundary_node_rejected','invalid_input_cavity_rejected'],
  'physical_admission':False,'native_steps':0,'script_sha256':sha(Path(__file__))}
 (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
def run(a):
 st=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 assert sha(a.input)==a.input_sha256,'input pin'
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v['points_world_m'];t=v['tetrahedra'].astype(np.int64);mat=v['material_indices']
 d,q=m.measurements(p,t);good=m.passes(d,q);assert np.all(d>0)
 source=np.loadtxt(a.source_node,skiprows=1,usecols=(1,2,3)).astype(np.float32)
 registered,counts=reg.registered_source_rows_from_map(a.registration_map,source)
 assert len(p)>=len(source)
 assert np.array_equal(p[:len(source)][registered].view(np.uint32),source[registered].view(np.uint32))
 b=top.boundary(t);bid=top.boundary_identity(b)
 fixed=np.zeros(len(p),bool);fixed[np.unique(b)]=True;fixed[:len(source)]|=registered
 flat=t.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 hits=np.bincount(t[~good].ravel(),minlength=len(p))
 nodes=np.flatnonzero((hits>0)&~fixed);nodes=nodes[np.lexsort((nodes,-hits[nodes]))]
 used=set();removed=[];added=[];newmat=[];ops=[];rej=collections.Counter();parentoffs=[0];newoffs=[0]
 existing={tuple(x) for x in np.sort(t,axis=1)}
 for node in nodes:
  if time.monotonic()-st>a.seconds or len(ops)>=a.maximum_operations:break
  rows=np.unique(order[offset[node]:offset[node+1]]//4)
  if any(int(x) in used for x in rows):rej['OVERLAPPING_OLD_CAVITY']+=1;continue
  if len(np.unique(mat[rows]))!=1:rej['MATERIAL_INTERFACE']+=1;continue
  old=t[rows];best=None
  for target in np.unique(old):
   if target==node:continue
   ans,why=propose(p,old,int(node),int(target))
   if ans is None:rej[why]+=1;continue
   nt,op=ans
   if any(tuple(x) in existing for x in np.sort(nt,axis=1)):
    rej['REPLACEMENT_ALREADY_EXISTS']+=1;continue
   value=(op['bad_after'],-op['minimum_normalized_det_after'],int(target))
   if best is None or value<best[0]:best=(value,nt,op)
  if best is None:continue
  _,nt,op=best
  # New neighbouring cavities may share a boundary, but never overlap old cells.
  used.update(map(int,rows));removed.extend(map(int,rows));added.extend(nt.tolist())
  newmat.extend([int(mat[rows[0]])]*len(nt));parentoffs.append(len(removed));newoffs.append(len(added))
  op['old_rows']=rows.tolist();ops.append(op)
 keep=np.ones(len(t),bool);keep[removed]=False
 nt=np.asarray(added,np.int64).reshape(-1,4);out=np.vstack([t[keep],nt]);outmat=np.r_[mat[keep],np.asarray(newmat,dtype=mat.dtype)]
 assert np.array_equal(top.boundary_identity(top.boundary(out)),bid)
 keys=np.sort(out,axis=1);assert len(np.unique(keys,axis=0))==len(out),'global duplicate'
 dd,qq=m.measurements(p,out);assert np.all(dd>0)
 expected=int(np.sum(~good))-sum(o['bad_before']-o['bad_after'] for o in ops)
 assert int(np.sum(~m.passes(dd,qq)))==expected
 v.update(tetrahedra=out,material_indices=outmat,retained_input_cell_rows=np.flatnonzero(keep),
  replacement_parent_offsets=np.asarray(parentoffs,np.int64),replacement_parent_cell_rows=np.asarray(removed,np.int64),
  replacement_tet_offsets=np.asarray(newoffs,np.int64),replacement_tetrahedra=nt)
 # Topology-dependent caches are invalidated only by their owning hash keys.
 output=a.output/'fixed-boundary-volume-proposal.npz';np.savez_compressed(output,**v)
 r={'status':'PASS_INFERRED_INTERIOR_NODE_RETIREMENT' if ops else 'NO_VALID_INTERIOR_NODE_RETIREMENT',
  'physical_admission':False,'native_steps':0,'input':str(a.input),'input_sha256':sha(a.input),
  'output':str(output),'output_sha256':sha(output),'script_sha256':sha(Path(__file__)),
  'registration_map':str(a.registration_map),'registration_map_sha256':sha(a.registration_map),
  'source_node':str(a.source_node),'source_node_sha256':sha(a.source_node),
  'registered_identity_counts':counts,'all_coordinates_bit_identical':True,'all_registered_anatomical_points_fixed':True,
  'oriented_boundary_exactly_unchanged':True,'exact_F32_volume_unchanged':True,'material_preserved':True,
  'input_cells':len(t),'output_cells':len(out),'bad_cells_before':int(np.sum(~good)),'bad_cells_after':expected,
  'passing_operations':len(ops),'eligible_bad_touching_interiors':len(nodes),'rejections':dict(rej),
  'operations':ops,'elapsed_s':time.monotonic()-st}
 (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({k:x for k,x in r.items() if k!='operations'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true')
 ap.add_argument('--input',type=Path);ap.add_argument('--input-sha256');ap.add_argument('--source-node',type=Path)
 ap.add_argument('--registration-map',type=Path);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--seconds',type=float,default=45);ap.add_argument('--maximum-operations',type=int,default=10000)
 a=ap.parse_args();fixture(a.output) if a.fixture else run(a)
