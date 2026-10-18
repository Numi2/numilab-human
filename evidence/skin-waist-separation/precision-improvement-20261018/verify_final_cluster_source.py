from pathlib import Path
from fractions import Fraction
import json,sys,numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.cardiac_cavity_intersections import _records,_audit_self_pairs_touching_faces
c=Path('/Users/n/numi-human-separation-20261018/C');d=c/'improve-wall23-final-008';old=_parse_payload((c/'improve-wall23-map-retired-004/resting-thorax.nhanatomy').read_bytes())[-1];rows=_parse_payload((d/'resting-thorax.nhanatomy').read_bytes())[-1]
for sid in old:
 for key in ('body_index','layer','flags'):assert old[sid][key]==rows[sid][key]
 if sid!=23:assert old[sid]['vertices6'].tobytes()==rows[sid]['vertices6'].tobytes() and np.array_equal(old[sid]['faces'],rows[sid]['faces'])
r=rows[23];v=r['vertices6'][:,:3];f=r['faces'];tri=v[f].astype(float);assert np.isfinite(v).all() and np.all(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>0)
assert len(np.unique(f))==len(v);assert len(np.unique(np.sort(f,axis=1),axis=0))==len(f)
e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);canon=np.sort(e,axis=1);unique,inverse,count=np.unique(canon,axis=0,return_inverse=True,return_counts=True);assert np.all(count==2);assert np.all(np.bincount(inverse,weights=np.where(e[:,0]<e[:,1],1,-1))==0)
n,_=connected_components(coo_matrix((np.ones(len(unique)),(unique[:,0],unique[:,1])),shape=(len(v),len(v))).tocsr(),directed=False);assert n==1;assert len(v)-len(unique)+len(f)==-2
keep=227812;selected=np.flatnonzero(np.any(f==keep,axis=1));patch=tri[selected];lo=patch.min(axis=(0,1));hi=patch.max(axis=(0,1));hits=np.flatnonzero(np.all(tri.max(axis=1)>=lo,axis=1)&np.all(tri.min(axis=1)<=hi,axis=1));vertices={int(i):tuple(Fraction(float(x)) for x in v[i]) for i in np.unique(f[hits])};records=_records(vertices,[tuple(map(int,t)) for t in f[hits]]);records=[(*row[:3],int(hits[i]),row[4]) for i,row in enumerate(records)];intersections=_audit_self_pairs_touching_faces(records,selected);assert intersections['count']==0
for center in np.unique(f[selected]):
 incident=f[np.any(f==center,axis=1)];link={}
 for face in incident:
  a,b=[int(x) for x in face if x!=center];link.setdefault(a,set()).add(b);link.setdefault(b,set()).add(a)
 assert all(len(x)==2 for x in link.values());seen={next(iter(link))};todo=list(seen)
 while todo:
  for x in link[todo.pop()]-seen:seen.add(x);todo.append(x)
 assert len(seen)==len(link)
report=dict(passed=True,scope='Changed wall source topology and exact self-intersections only; no whole-body motion/interdomain anatomical admission',all_other_524_rows_geometry_exact=True,all_stable_identity_body_layer_flags_exact=True,vertices=len(v),faces=len(f),closed_oriented_two_face_edges=len(unique),face_connected_components=n,euler_characteristic=-2,all_allocated_vertices_referenced=True,no_duplicate_or_zero_area_faces=True,changed_star_vertex_links_are_single_cycles=True,changed_star_self_intersections=intersections)
(d/'source-quality-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
