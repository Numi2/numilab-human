"""Independent source-decimal geometry audit; no model or physical stepping.
Exact source decimals are scaled to integers for triangle predicates and moments.
Winding values are supplemental floating witnesses; permanent nesting admission
uses the separate exact Float64-coordinate ray-parity checker.
"""
from pathlib import Path
import hashlib,json
_lock=json.loads(Path("sources.lock.json").read_text())
_archive=Path("Sources/partof_BP3D_4.0_obj_99.zip")
assert hashlib.sha256(_archive.read_bytes()).hexdigest()==_lock["sources"]["bodyparts3d_4"]["files"][_archive.name]["sha256"]=="9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97"
_initial=Path("tools/cvsim21_reference/evidence/20260912/original-initial.json")
assert hashlib.sha256(_initial.read_bytes()).hexdigest()=="fad558482f8bfeb94d531d4538e29d7a2c8a59ea63f32496467b087bcb343ff7"
from collections import defaultdict,Counter
from decimal import Decimal
from fractions import Fraction
import json,zipfile
z=zipfile.ZipFile('Sources/partof_BP3D_4.0_obj_99.zip');scale=10**8
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def topo(v,f):
 edges=defaultdict(list);adj=[set() for _ in f]
 for i,t in enumerate(f):
  for a,b in zip(t,t[1:]+t[:1]):edges[tuple(sorted((a,b)))].append((i,a,b))
 for rows in edges.values():
  for r in rows:adj[r[0]].update(s[0] for s in rows if s[0]!=r[0])
 rem=set(range(len(f)));sizes=[]
 while rem:
  start=min(rem);seen={start};stack=[start];rem.remove(start)
  while stack:
   for j in adj[stack.pop()] & rem:rem.remove(j);seen.add(j);stack.append(j)
  sizes.append(len(seen))
 boundary=[e for e,r in edges.items() if len(r)==1];ba=defaultdict(set)
 for a,b in boundary:ba[a].add(b);ba[b].add(a)
 rem=set(ba);bs=[]
 while rem:
  a=min(rem);seen={a};stack=[a];rem.remove(a)
  while stack:
   for b in ba[stack.pop()] & rem:rem.remove(b);seen.add(b);stack.append(b)
  bs.append((len(seen),sum(len(ba[x]) for x in seen)//2,dict(Counter(len(ba[x]) for x in seen))))
 return dict(V=len(v),E=len(edges),F=len(f),euler=len(v)-len(edges)+len(f),face_components=sorted(sizes,reverse=True),boundary_edges=len(boundary),boundary_components=bs,nonmanifold_edges=sum(len(r)>2 for r in edges.values()),orientation_conflicts=sum(len(r)==2 and r[0][1:]==r[1][1:] for r in edges.values()))
meshes={}
for name,label,cvindex in [('FJ2424','RA',15),('FJ2423','RV',16),('FJ2425','LA',19),('FJ2422','LV',20)]:
 lines=z.read('partof_BP3D_4.0_obj_99/'+name+'.obj').decode().splitlines();v=[];f=[]
 for line in lines:
  x=line.split()
  if x and x[0]=='v':
   assert len(x)==4
   c=tuple(Decimal(q)*scale for q in x[1:]);assert all(q==int(q) for q in c);v.append(tuple(map(int,c)))
  elif x and x[0]=='f':assert len(x)==4;f.append(tuple(int(q.split('/')[0])-1 for q in x[1:]))
 u=[];lookup={};mapping=[]
 for c in v:
  if c not in lookup:lookup[c]=len(u);u.append(c)
  mapping.append(lookup[c])
 qf=[tuple(mapping[i] for i in t) for t in f]
 sixV=sum(dot(u[a],cross(u[b],u[c])) for a,b,c in qf)
 center=tuple((min(c[k] for c in u)+max(c[k] for c in u))//2 for k in range(3))
 translated=[sub(c,center) for c in u]
 shifted=sum(dot(translated[a],cross(translated[b],translated[c])) for a,b,c in qf)
 assert sixV==shifted
 num=[0,0,0]
 for a,b,c in qf:
  d=dot(u[a],cross(u[b],u[c]))
  for k in range(3):num[k]+=d*(u[a][k]+u[b][k]+u[c][k])
 centroid=[float(Fraction(x,4*sixV*scale)) for x in num];vol=float(Fraction(sixV,6*scale**3*1000))
 badlinks=[]
 for vi in range(len(u)):
  link=defaultdict(set)
  for face in qf:
   if vi in face:
    a,b=[i for i in face if i!=vi];link[a].add(b);link[b].add(a)
  seen=set();stack=[next(iter(link))]
  while stack:
   x=stack.pop()
   if x not in seen:seen.add(x);stack.extend(link[x]-seen)
  if len(seen)!=len(link) or any(len(a)!=2 for a in link.values()):badlinks.append(vi)
 init=json.load(open('tools/cvsim21_reference/evidence/20260912/original-initial.json'))['volume_mL'][cvindex]
 result=dict(id=name,chamber=label,raw=topo(v,f),quotient=topo(u,qf),collapsed_faces=sum(len(set(t))!=3 for t in qf),zero_area_faces=sum(cross(sub(u[b],u[a]),sub(u[c],u[a]))==(0,0,0) for a,b,c in qf),bad_vertex_links=badlinks,signed_volume_mL=vol,centroid_mm=centroid,source_declared_volume=[x for x in lines if x.startswith('# Volume')],exact_origin_translation_volume_invariant=sixV==shifted,cvsim_initial_volume_mL=init,cvsim_to_bp3d_volume_ratio=init/vol)
 print(json.dumps(result))
 meshes[label]=(u,qf)


from bisect import bisect_right
import itertools,math,time
def orient2(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def in2(p,t):
 s=[orient2(t[i],t[(i+1)%3],p) for i in range(3)]
 return min(s)>=0 or max(s)<=0
def sameplane_intersections(A,B,n):
 axes=[i for i in range(3) if i!=max(range(3),key=lambda k:abs(n[k]))]
 proj=lambda p:tuple(p[k] for k in axes)
 a,b=list(map(proj,A)),list(map(proj,B));out=[]
 for i,p in enumerate(a):
  if in2(p,b):out.append(A[i])
 for i,p in enumerate(b):
  if in2(p,a):out.append(B[i])
 for i in range(3):
  x,y=a[i],a[(i+1)%3]
  for j in range(3):
   c,d=b[j],b[(j+1)%3];o1,o2=orient2(c,d,x),orient2(c,d,y);p1,p2=orient2(x,y,c),orient2(x,y,d)
   if o1*o2<0 and p1*p2<0:
    t=Fraction(o1,o1-o2);out.append(tuple(A[i][k]+t*(A[(i+1)%3][k]-A[i][k]) for k in range(3)))
 return out
def segmenttri(P,Q,T,n):
 dp=dot(n,sub(P,T[0]));dq=dot(n,sub(Q,T[0]))
 if (dp>0 and dq>0) or (dp<0 and dq<0) or dp==dq:return []
 den=dp-dq;num=tuple(dp*Q[k]-dq*P[k] for k in range(3))
 if den<0:den=-den;num=tuple(-x for x in num)
 axes=[i for i in range(3) if i!=max(range(3),key=lambda k:abs(n[k]))];x,y=axes;signs=[]
 for i in range(3):
  a,b=T[i],T[(i+1)%3];signs.append((b[x]-a[x])*(num[y]-den*a[y])-(b[y]-a[y])*(num[x]-den*a[x]))
 if min(signs)<0 and max(signs)>0:return []
 return [tuple(Fraction(v,den) for v in num)]
def intersects(A,B):
 na=cross(sub(A[1],A[0]),sub(A[2],A[0]));nb=cross(sub(B[1],B[0]),sub(B[2],B[0]))
 sa=[dot(na,sub(p,A[0])) for p in B];sb=[dot(nb,sub(p,B[0])) for p in A]
 if min(sa)>0 or max(sa)<0 or min(sb)>0 or max(sb)<0:return []
 if all(x==0 for x in sa):return sameplane_intersections(A,B,na)
 out=[]
 for i in range(3):out.extend(segmenttri(A[i],A[(i+1)%3],B,nb));out.extend(segmenttri(B[i],B[(i+1)%3],A,na))
 return out
def records(mesh):
 v,f=mesh
 return [(tuple(v[k] for k in t),tuple(min(v[k][j] for k in t) for j in range(3)),tuple(max(v[k][j] for k in t) for j in range(3)),i,t) for i,t in enumerate(f)]
def auditpair(first,second,selftest):
 B=sorted(second,key=lambda r:r[1][0]);starts=[r[1][0] for r in B];candidates=0;hits=[];expected=0
 for A,lo,hi,i,ids in first:
  for T,blo,bhi,j,bids in B[:bisect_right(starts,hi[0])]:
   if selftest and j<=i:continue
   if any(hi[k]<blo[k] or bhi[k]<lo[k] for k in range(3)):continue
   candidates+=1;points=intersects(A,T)
   if not points:continue
   common=set(A)&set(T)
   def expected_point(p):
    if p in common:return True
    if len(common)==2:
     c,d=tuple(common);return cross(sub(p,c),sub(d,c))==(0,0,0) and all(min(c[k],d[k])<=p[k]<=max(c[k],d[k]) for k in range(3))
    return False
   if selftest and all(expected_point(p) for p in points):expected+=1
   else:hits.append((i,j,len(common),[[float(q/scale) for q in p] for p in points[:2]]))
 return dict(bbox_pairs=candidates,expected_shared_vertex_or_edge_pairs=expected,unexpected_intersection_count=len(hits),first_intersections=hits[:8])
rs={name:records(mesh) for name,mesh in meshes.items()}
for name in meshes:print('SELF',name,json.dumps(auditpair(rs[name],rs[name],True)))
for a,b in itertools.combinations(meshes,2):print('CROSS',a,b,json.dumps(auditpair(rs[a],rs[b],False)))
def winding(p,mesh):
 v,f=mesh;terms=[]
 for a,b,c in f:
  A,B,C=[tuple((v[i][k]-p[k])/scale for k in range(3)) for i in (a,b,c)];la,lb,lc=[math.sqrt(dot(q,q)) for q in (A,B,C)]
  terms.append(2*math.atan2(dot(A,cross(B,C)),la*lb*lc+dot(A,B)*lc+dot(B,C)*la+dot(C,A)*lb))
 return math.fsum(terms)/(4*math.pi)
for a,b in itertools.combinations(meshes,2):print('CONTAINMENT_WINDING',a,b,winding(meshes[a][0][0],meshes[b]),winding(meshes[b][0][0],meshes[a]))

