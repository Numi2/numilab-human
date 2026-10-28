#!/usr/bin/env python3
"""Local connectivity repair for Float32 slivers; no vertex movement or physics."""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time,types
import numpy as np
TOOLS=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/tools')
pkg=types.ModuleType('numi_tools');pkg.__path__=[str(TOOLS)];sys.modules['numi_tools']=pkg
for name,attrs in [('cardiac_cavity_geometry',{'analyze_topology':lambda *a,**k:None}),('model',{'ImportError':RuntimeError}),('physiology',{'canonical':lambda x:x})]:
 m=types.ModuleType('numi_tools.'+name);m.__dict__.update(attrs);sys.modules[m.__name__]=m
from numi_tools import cardiac_cavity_intersections as exact
def edgekey(a,b):return (min(int(a),int(b)),max(int(a),int(b)))
def nrm(P,f):return np.cross(P[f[1]]-P[f[0]],P[f[2]]-P[f[0]])
def quality(P,f):
 q=P[f];nn=np.linalg.norm(np.cross(q[1]-q[0],q[2]-q[0]))
 lengths=[np.linalg.norm(q[(k+1)%3]-q[k]) for k in range(3)]
 return nn/max(lengths) if max(lengths)>0 else 0.
class Flips:
 def __init__(self,P,F,parents=None):
  self.P64=np.asarray(P,np.float64);self.P=self.P64.astype(np.float32).astype(float);self.F=np.asarray(F,np.int64).copy()
  self.lattice=[exact.float32_point_lattice_key(p)for p in self.P]
  self.edges=collections.defaultdict(set)
  self.parents=([set([i])for i in range(len(F))] if parents is None else [set(map(int,p)) for p in parents])
  if len(self.parents)!=len(F):raise ValueError("face ancestry length mismatch")
  for i,f in enumerate(self.F):
   for k in range(3):self.edges[edgekey(f[k],f[(k+1)%3])].add(i)
  self.recs=[self.record(i,f)for i,f in enumerate(self.F)]
  Q=self.P[self.F];self.lo=Q.min(1);self.hi=Q.max(1);self.edits=[]
 def record(self,i,f):
  tri=tuple(self.lattice[v]for v in f)
  if not any(exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))):return None
  return (tri,tuple(min(v[k]for v in tri)for k in range(3)),tuple(max(v[k]for v in tri)for k in range(3)),i,tuple(map(int,f)))
 def pairs(self,changed,newfaces=None):
  changed=set(changed);cand=[]
  for i in sorted(changed):
   f=self.F[i]if newfaces is None else newfaces[i]
   rec=self.record(i,f)
   if rec is None:return None
   q=self.P[f];lo=q.min(0);hi=q.max(0)
   nearby=np.flatnonzero(np.all(self.hi>=lo,axis=1)&np.all(self.lo<=hi,axis=1))
   for j in sorted(set(map(int,nearby))|changed):
    j=int(j)
    if j==i or (j in changed and j<i):continue
    other=self.record(j,newfaces[j])if newfaces is not None and j in changed else self.recs[j]
    if other is None:continue
    cand.append((rec,other)if i<j else(other,rec))
  return len(exact._audit_record_pairs(cand,same_surface=True)['triangle_pairs'])
 def try_flip(self,e,require_clear=True):
  fs=sorted(self.edges[e])
  if len(fs)!=2:return False
  i,j=fs;a,b=e;f=self.F[i];g=self.F[j]
  # Canonical directed edge must occur in the first face, opposite in second.
  fi=next((k for k in range(3)if f[k]==a and f[(k+1)%3]==b),None)
  if fi is None:a,b=b,a;fi=next((k for k in range(3)if f[k]==a and f[(k+1)%3]==b),None)
  if fi is None or not any(g[k]==b and g[(k+1)%3]==a for k in range(3)):return False
  c=next(int(x)for x in f if x!=a and x!=b);d=next(int(x)for x in g if x!=a and x!=b)
  if c==d or self.edges.get(edgekey(c,d)):return False
  nf=np.array([c,d,b]);ng=np.array([d,c,a]);new={i:nf,j:ng}
  oldq=min(quality(self.P,f),quality(self.P,g));newq=min(quality(self.P,nf),quality(self.P,ng))
  if not np.isfinite(newq)or newq<=max(oldq*1.05,1e-16):return False
  guide=nrm(self.P64,f)+nrm(self.P64,g)
  if np.linalg.norm(guide)==0:return False
  if any(np.dot(nrm(self.P,h),guide)<=0 or np.dot(nrm(self.P64,h),guide)<=0 for h in (nf,ng)):return False
  before=self.pairs(fs)
  after=self.pairs(fs,new)
  if after is None or (require_clear and after!=0)or (before is not None and after>before):return False
  original=[self.F[i].tolist(),self.F[j].tolist()];par=self.parents[i]|self.parents[j]
  for row in fs:
   old=self.F[row]
   for k in range(3):
    key=edgekey(old[k],old[(k+1)%3]);self.edges[key].remove(row)
    if not self.edges[key]:del self.edges[key]
  for row,face in new.items():
   self.F[row]=face;self.parents[row]=set(par);self.recs[row]=self.record(row,face)
   q=self.P[face];self.lo[row]=q.min(0);self.hi[row]=q.max(0)
   for k in range(3):self.edges[edgekey(face[k],face[(k+1)%3])].add(row)
  self.edits.append({'faces':fs,'old_corners':original,'new_corners':[nf.tolist(),ng.tolist()],
    'old_min_altitude_m':oldq,'new_min_altitude_m':newq,'old_exact_touching_pairs':before,
    'new_exact_touching_pairs':after,'source_face_ancestry':sorted(par)})
  return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--face-rows',type=Path)
 ap.add_argument('--attempts',type=int,default=256);ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--self-test',action='store_true')
 a=ap.parse_args()
 if a.self_test:
  P=np.array([[0,0,0],[1,0,0],[.5,1e-46,0],[.5,-1,0]],float);F=np.array([[0,1,2],[1,0,3]])
  f=Flips(P,F);boundary={e for e,v in f.edges.items()if len(v)==1}
  assert f.recs[0]is None;assert f.try_flip((0,1));assert f.pairs([0,1])==0
  assert {e for e,v in f.edges.items()if len(v)==1}==boundary
  assert np.array_equal(f.P64,P);assert all(quality(f.P,t)>.4 for t in f.F)
  print('local_Float32_edge_flip=PASS positions_unchanged=1 exact_new_pairs=0 boundary_edges_preserved=1');return
 if not a.input or not a.out:raise ValueError('input and out required')
 z=np.load(a.input,allow_pickle=False);data={k:z[k]for k in z.files};P=data['points_world_m'];F=data['faces']
 f=Flips(P,F);start=time.monotonic();bad=set(np.flatnonzero([r is None for r in f.recs]).tolist())
 if a.face_rows:bad.update(map(int,np.load(a.face_rows,allow_pickle=False).ravel()))
 attempts=0;passes=0
 for _ in range(4):
  progress=0
  for row in sorted(bad,key=lambda x:quality(f.P,f.F[x])):
   candidates=sorted((edgekey(f.F[row][k],f.F[row][(k+1)%3])for k in range(3)),
     key=lambda e:-np.linalg.norm(f.P[e[0]]-f.P[e[1]]))
   for e in candidates:
    if attempts>=a.attempts or time.monotonic()-start>=a.seconds:break
    attempts+=1
    if f.try_flip(e):progress+=1;break
   if attempts>=a.attempts or time.monotonic()-start>=a.seconds:break
  passes+=1
  if not progress or attempts>=a.attempts or time.monotonic()-start>=a.seconds:break
 a.out.mkdir(parents=True,exist_ok=False);data['faces']=f.F;data['edge_flip_parent_offsets']=np.r_[0,np.cumsum([len(p)for p in f.parents])].astype(np.int64)
 data['edge_flip_parent_face_rows']=np.array([j for p in f.parents for j in sorted(p)],np.int64)
 changed=np.array(sorted({i for edit in f.edits for i in edit['faces']}),np.int64)
 if len(changed):
  if 'face_source_row'in data:data['face_source_row']=data['face_source_row'].copy();data['face_source_row'][changed]=-1
  if 'face_source_role'in data:data['face_source_role']=data['face_source_role'].copy();data['face_source_role'][changed]=255
 np.savez_compressed(a.out/'candidate.npz',**data)
 result={'scope':'local connectivity candidate; requires full F64/F32 self/clearance/thickness/cell gates before physical admission',
  'input':str(a.input),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256((a.out/'candidate.npz').read_bytes()).hexdigest(),
  'attempts':attempts,'passes':passes,'accepted_flips':len(f.edits),'emitted_collapsed_faces':sum(r is None for r in f.recs),
  'vertices_moved':0,'elapsed_s':time.monotonic()-start,'edits':f.edits}
 (a.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='edits'}))
if __name__=='__main__':main()
