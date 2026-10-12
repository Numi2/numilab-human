#!/usr/bin/env python3
import hashlib, importlib.util, json, sys, os, math
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.set_int_max_str_digits(0)
ROOT=Path('/Users/n/numi-human-back-m6-1349')
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
CAPS={
 '007':A/'back-native-014-raw-parent-007/native-back-audit-001',
 '009':A/'back-native-014-009-60s/native-back-audit-001',
 '016':A/'back-forearm-native-014-016-back412-right177/native-back412-audit-002',
 '017':A/'back-forearm-native-014-017-back412-right177-60s/native-back412-common-capture-audit-001',
}
BASE=ROOT/'attempt-408-row157-native7967-subresolution-edge-repair-001/full_candidate.json'
CAND=ROOT/'attempt-412-descending-cap-self-crossing-flip-screen-003/full_candidate.json'
OUT=ROOT/'attempt-417-late-native-midline-contact-audit-001'; OUT.mkdir(exist_ok=True)
HEAD=Path('/Users/n/numi-human-exact-normalization-m6-1348')
sys.path.insert(0,str(HEAD/'src'))
from numilab_human import cardiac_cavity_intersections as ci

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse_candidate(p):
 d=json.loads(Path(p).read_text())
 pts={int(s):[tuple(F(int(x[0]),int(x[1])) for x in q) for q in vals] for s,vals in d['points_exact'].items()}
 faces={int(s):[None if f is None else tuple(map(int,f)) for f in vals] for s,vals in d['faces'].items()}
 ancestry={int(s):vals for s,vals in d['face_ancestry'].items()}
 return d,pts,faces,ancestry

def barycentric(p,tri):
 d1=ci._sub(tri[1],tri[0]);d2=ci._sub(tri[2],tri[0]);dp=ci._sub(p,tri[0]);n=ci._cross(d1,d2)
 drop=max(range(3),key=lambda i:abs(n[i])); axes=[i for i in range(3) if i!=drop];i,j=axes
 det=d1[i]*d2[j]-d1[j]*d2[i]
 if not det: raise ValueError('degenerate projection')
 u=(dp[i]*d2[j]-dp[j]*d2[i])/det
 v=(d1[i]*dp[j]-d1[j]*dp[i])/det
 return (1-u-v,u,v)
def xclass(xs):
 if all(q==0 for q in xs):return 'all_x0'
 if min(xs)<0<max(xs):return 'straddles_x0'
 if any(q==0 for q in xs):return 'touches_x0'
 return 'negative' if max(xs)<0 else 'positive'
def dist(a,b):return math.sqrt(sum((float(a[i]-b[i])/2**149)**2 for i in range(3)))
def audit_pair(ra,rb,pts,sourcefaces,ancestry,world,nativefaces,face_slots,same):
 recA=ci._records(world[ra],nativefaces[ra]);recB=recA if ra==rb else ci._records(world[rb],nativefaces[rb])
 audit=ci._audit_pair(recA,recB,same_surface=same)
 cats=Counter(); signcats=Counter(); segs=[]; examples=[]; source_away=[]; angle=Counter();dimensions=Counter();interface_support=0
 for ia,ib in audit['triangle_pairs']:
  pa=recA[ia][0];pb=recB[ib][0]
  ip=sorted(set(ci.triangle_intersection_points(pa,pb)))
  dim='point' if len(ip)==1 else 'edge' if len(ip)==2 else 'area' if len(ip)>=3 else 'empty'
  dimensions[dim]+=1
  slot_a=face_slots[ra][ia];slot_b=face_slots[rb][ib]
  fia=sourcefaces[ra][slot_a];fib=sourcefaces[rb][slot_b]
  xsa=[pts[ra][v][0] for v in fia];xsb=[pts[rb][v][0] for v in fib]
  xa=[];xb=[]
  for p in ip:
   ba=barycentric(p,pa);bb=barycentric(p,pb)
   xa.append(sum((ba[k]*pts[ra][fia[k]][0] for k in range(3)),F(0)))
   xb.append(sum((bb[k]*pts[rb][fib[k]][0] for k in range(3)),F(0)))
  maxx=max([abs(x) for x in xa+xb],default=F(0))
  supported=all(x==0 for x in xa+xb)
  interface_support+=int(supported)
  source_away.append(float(maxx))
  ma=ancestry[ra][slot_a];mb=ancestry[rb][slot_b]
  taga=(int(ma.get('origin_kind',-1)),int(ma.get('partition_role',-1)),int(ma.get('parent_stable_id',0)),xclass(xsa))
  tagb=(int(mb.get('origin_kind',-1)),int(mb.get('partition_role',-1)),int(mb.get('parent_stable_id',0)),xclass(xsb))
  tags=tuple(sorted((taga,tagb)))
  cats[(dim,'source_x0' if supported else 'off_source_x0',tags)]+=1
  def sxclass(values):
   if all(v==0 for v in values):return 'zero'
   if all(v<=0 for v in values):return 'nonpositive'
   if all(v>=0 for v in values):return 'nonnegative'
   return 'mixed'
  signcats[(dim,sxclass(xa),sxclass(xb))]+=1
  na=ci._cross(ci._sub(pa[1],pa[0]),ci._sub(pa[2],pa[0]));nb=ci._cross(ci._sub(pb[1],pb[0]),ci._sub(pb[2],pb[0]))
  angle['parallel' if not any(ci._cross(na,nb)) else 'nonparallel']+=1
  if dim=='edge':segs.append(dist(ip[0],ip[1]))
  if len(examples)<100:
   examples.append({'faces':[ia,ib],'source_faces':[[ra,ia,ma],[rb,ib,mb]],'dimension':dim,'source_intersection_x_m':[[str(x) for x in xa],[str(x) for x in xb]],'posed_intersection_m':[[float(v)/2**149 for v in p] for p in ip]})
 return {'raw_pair_count':len(audit['triangle_pairs']),'aabb_candidates':audit['aabb_candidate_pairs'],'allowed_index_contacts':audit['allowed_shared_vertex_or_edge_pairs'],'dimensions':dict(dimensions),'source_strata':{str(k):v for k,v in cats.items()},'source_x_sign_pair_counts':{str(k):v for k,v in signcats.items()},'plane_relation':dict(angle),'all_intersections_exactly_on_source_x0_count':interface_support,'off_source_x0_intersection_count':len(audit['triangle_pairs'])-interface_support,'max_source_abs_x_over_all_intersections_m':max(source_away,default=0.0),'max_source_abs_x_by_point_m':sorted(source_away,reverse=True)[:20],'edge_segment_length_m':{'count':len(segs),'min':min(segs) if segs else None,'median':float(np.median(segs)) if segs else None,'max':max(segs) if segs else None},'examples':examples}

def main():
 bd,bpts,bfaces,banc=parse_candidate(BASE);cd,pts,faces,anc=parse_candidate(CAND)
 capdir=CAPS[sys.argv[1]];step=int(sys.argv[2]);rows={};world={};nativefaces={};face_slots={}
 load_rows=[157,158] if sys.argv[1] in ('016','017') else [153,154,155,156,157,158]
 for s in load_rows:
  z=np.load(capdir/f'step-{step}-row-{s}.npz',allow_pickle=False);pos=np.asarray(z['positions'],dtype='<f4');cf=np.asarray(z['faces'],dtype=np.int64)
  live=[i for i,f in enumerate(faces[s]) if f is not None]
  if len(live)!=len(cf):raise RuntimeError(f'face count mismatch row{s}: {len(live)} != {len(cf)}')
  if sys.argv[1] in ('016','017'):
   ff=[tuple(map(int,f)) for f in cf]
   expected=[tuple(map(int,faces[s][i])) for i in live]
   if ff!=expected:raise RuntimeError(f'actual native face order differs from candidate412 row{s}')
  else:
   base_live=[i for i,f in enumerate(bfaces[s]) if f is not None]
   if base_live!=live:raise RuntimeError(f'base/candidate live slot mapping differs row{s}')
   vmap={}
   for fi,nf in zip(live,cf):
    for cv,nv in zip(bfaces[s][fi],nf):
     if cv in vmap and vmap[cv]!=int(nv):raise RuntimeError(f'vertex map conflict row{s} vertex{cv}')
     vmap[cv]=int(nv)
    if tuple(vmap[v] for v in bfaces[s][fi])!=tuple(map(int,nf)):raise RuntimeError('native face mismatch')
   if len(vmap)!=len({v for f in bfaces[s] if f is not None for v in f}):raise RuntimeError('incomplete vmap')
   ff=[tuple(vmap[v] for v in f) for f in faces[s] if f is not None]
  fa=live
  world[s]=[ci.float32_point_lattice_key(tuple(float(x) for x in q)) for q in pos]
  # Face-index order is candidate local slot order; all candidate412 rows must retain live ordering.
  nativefaces[s]=ff
  face_slots[s]=fa
  rows[s]={'npz':str(capdir/f'step-{step}-row-{s}.npz'),'npz_sha256':sha(capdir/f'step-{step}-row-{s}.npz'),'faces':ff,'live_slots':fa}
 # Pairwise source-plane diagnostics use the full candidate; joined-boundary self checks
 # then reproduce attempt414's explicit cancellation of only declared paired facets.
 results={}
 checks=[(157,158,False),(157,157,True),(158,158,True)] if sys.argv[1] in ('016','017') else [(153,154,False),(155,156,False),(157,158,False),(157,157,True),(158,158,True)]
 for a,b,same in checks:
  name=f'{a}+{b}' if a!=b else f'{a}-self-full'
  results[name]=audit_pair(a,b,pts,faces,anc,world,nativefaces,face_slots,same)
 joined=json.loads((ROOT/'attempt-414-sixrow-joined-exterior-audit-001/joined-boundary-face-ancestry.json').read_text())
 selected={s:set() for s in [157,158]}
 for rr,fi in joined['union_faces']:
  if rr in selected:selected[rr].add(int(fi))
 jnative={s:list(nativefaces[s]) for s in nativefaces};jslots={s:list(face_slots[s]) for s in face_slots}
 for s in [157,158]:
  keep=[i for i,slot in enumerate(jslots[s]) if slot in selected[s]]
  jnative[s]=[jnative[s][i] for i in keep];jslots[s]=[jslots[s][i] for i in keep]
  if len(jslots[s])!=len(selected[s]):raise RuntimeError(f'joined face selection mismatch row{s}')
  results[f'{s}-self-joined-boundary']=audit_pair(s,s,pts,faces,anc,world,jnative,jslots,True)
 out={'schema':'numi.human.trapezius.attempt416.midline-contact-support.v1','candidate_admission':False,'native_admission':False,'audit_script_path':str(Path(__file__)),'audit_script_sha256':sha(__file__),'candidate_path':str(ROOT/'attempt-412-descending-cap-self-crossing-flip-screen-003/full_candidate.json'),'candidate_sha256':sha(CAND),'base_candidate_path':str(BASE),'base_candidate_sha256':sha(BASE),'capture':{'grid':sys.argv[1],'step':step,'root':str(capdir),'rows':{str(s):{'path':rows[s]['npz'],'sha256':rows[s]['npz_sha256']} for s in rows}},'intersection_module':str(HEAD/'src/numilab_human/cardiac_cavity_intersections.py'),'intersection_module_sha256':sha(HEAD/'src/numilab_human/cardiac_cavity_intersections.py'),'results':results,'interpretation':'Exact intersections are classified by barycentrically interpolating each source triangle x coordinate at exact intersections of the actual posed F32 triangles. This is a source-triangle localization diagnostic, not an inverse of nonlinear skinning and not a signed physical clearance. A zero source-x range supports localization to the declared x=0 cap/interface; it does not waive topology or by itself prove a shared material face. No volumetric overlap was computed.'}
 outp=OUT/f'grid-{sys.argv[1]}-step-{step}.json';outp.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({'report':str(outp),'sha256':sha(outp),'results':{k:{x:v[x] for x in ['raw_pair_count','dimensions','plane_relation','all_intersections_exactly_on_source_x0_count','off_source_x0_intersection_count','max_source_abs_x_over_all_intersections_m','edge_segment_length_m']} for k,v in results.items()}},sort_keys=True))
if __name__=='__main__':main()
