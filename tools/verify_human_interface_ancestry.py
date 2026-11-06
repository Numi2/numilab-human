"""Verify bounded I diagonal changes inside the existing typed-role compiler.

Retain the frozen exporter and every coordinate, volume, source and mass check.
A single parent-face row is only a representative; both full ancestry CSRs must
replay the exact union of the two authenticated immediate parent faces.
"""
import collections
import numpy as np


def edge_chain(faces):
 out=collections.Counter()
 for a,b,c in np.asarray(faces):
  for u,v in [(a,b),(b,c),(c,a)]:
   u,v=int(u),int(v);out[tuple(sorted((u,v)))]+=1 if u<v else -1
 return {k:v for k,v in out.items() if v}


def diagonal_gap(points,old,new):
 old=np.asarray(old);new=np.asarray(new)
 nodes=np.unique(old)
 if old.shape!=(2,3) or new.shape!=(2,3) or len(nodes)!=4 or not np.array_equal(nodes,np.unique(new)) or edge_chain(old)!=edge_chain(new):
  raise ValueError('interface flip changes its oriented quad boundary')
 oldedge=np.intersect1d(old[0],old[1]);newedge=np.intersect1d(new[0],new[1])
 if len(oldedge)!=2 or len(newedge)!=2 or len(np.intersect1d(oldedge,newedge)):
  raise ValueError('interface flip does not exchange the two quad diagonals')
 x=points[nodes].astype(float);_,_,basis=np.linalg.svd(x-x.mean(0),full_matrices=False)
 q={int(i):(points[i].astype(float)-x.mean(0))@basis[:2].T for i in nodes}
 u=q[int(oldedge[1])]-q[int(oldedge[0])];v=q[int(newedge[1])]-q[int(newedge[0])]
 try:ts=np.linalg.solve(np.column_stack([u,-v]),q[int(newedge[0])]-q[int(oldedge[0])])
 except np.linalg.LinAlgError as exc:raise ValueError('singular interface quad') from exc
 if not np.isfinite(ts).all() or np.any(ts<=0) or np.any(ts>=1):raise ValueError('nonconvex projected interface quad')
 for faces in [old,new]:
  tri=points[faces].astype(float)
  normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
  if np.any(np.linalg.norm(normal,axis=1)==0) or np.dot(normal[0],normal[1])<=0:raise ValueError('folded interface quad')
 a=points[oldedge].astype(float);b=points[newedge].astype(float)
 return float(np.linalg.norm(a[0]+ts[0]*(a[1]-a[0])-b[0]-ts[1]*(b[1]-b[0])))


def validate_ancestry(payload,offset,lanes):
 off=np.asarray(payload[offset]);n=len(payload['faces'])
 if off.shape!=(n+1,) or off.dtype.kind not in 'iu' or off[0]!=0 or np.any(np.diff(off)<=0):raise ValueError('invalid interface ancestry offsets')
 values=[np.asarray(payload[k]) for k in lanes]
 if any(v.ndim!=1 or len(v)!=off[-1] or v.dtype.kind not in 'iu' for v in values):raise ValueError('invalid interface ancestry lanes')


def ancestry_rows(payload,row,offset,lanes):
 off=payload[offset]
 return np.unique(np.column_stack([payload[k][off[row]:off[row+1]] for k in lanes]),axis=0)


def verify_interface_replay(child,parent,quotient,rows,maximum_gap_m=.000020):
 # NPZ fields are compressed; load each participating lane once per check.
 if hasattr(child,'files'):child={k:child[k] for k in child.files}
 if hasattr(parent,'files'):parent={k:parent[k] for k in parent.files}
 if not 0<maximum_gap_m<=.000020:raise ValueError('invalid interface construction bound')
 expected=quotient[parent['faces'][rows]];actual=child['faces'];points=child['points_world_m_f32']
 if expected.shape!=actual.shape:raise ValueError('interface face domain changed')
 changed=np.flatnonzero(np.any(expected!=actual,axis=1))
 if not len(changed):return dict(patches=0,maximum_gap_m=0.)
 # This extension handles connected face-diagonal changes, never point motion.
 if not np.array_equal(quotient,np.arange(len(points))) or not np.array_equal(points,parent['points_world_m_f32']):raise ValueError('interface repair changes point identity')
 groups=[('face_ancestry_offsets',['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows']),('quality_star_patch_union_offsets',['quality_star_patch_union_face_source_'+k for k in ['layer','body','bone','face_id','original_id','identity0','identity1']])]
 for payload in [child,parent]:
  for group in groups:validate_ancestry(payload,*group)
 batches=collections.defaultdict(list)
 for row in changed:
  signature=tuple(ancestry_rows(child,int(row),*groups[0]).ravel());batches[signature].append(int(row))
 largest=0.
 for patch in batches.values():
  if len(patch)!=2:raise ValueError('interface ancestry does not identify exactly two changed faces')
  gap=diagonal_gap(points,expected[patch],actual[patch])
  if not np.isfinite(gap) or gap>maximum_gap_m:raise ValueError('interface diagonal exceeds construction bound')
  largest=max(largest,gap)
  for offset,lanes in groups:
   union=np.unique(np.concatenate([ancestry_rows(parent,int(rows[i]),offset,lanes) for i in patch]),axis=0)
   for i in patch:
    if not np.array_equal(ancestry_rows(child,i,offset,lanes),union):raise ValueError('interface repair drops or fabricates a parent ancestry')
 return dict(patches=len(batches),maximum_gap_m=largest)
