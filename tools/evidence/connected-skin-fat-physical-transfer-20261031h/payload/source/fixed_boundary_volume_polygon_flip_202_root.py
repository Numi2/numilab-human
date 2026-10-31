"""Remove closed interior edge stars with quality-selected polygon triangulations."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np,json,hashlib,time,argparse,importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cavity198',H/'fixed_boundary_volume_flip_198_root.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def ring_order(old,d,e):
 links=[tuple(sorted(set(map(int,tet))-set((d,e)))) for tet in old]
 if any(len(link)!=2 for link in links) or len(set(links))!=len(links):return None
 adj={}
 for a,b in links:adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
 if len(adj)!=len(old) or any(len(v)!=2 for v in adj.values()):return None
 first=min(adj);ring=[first];prev=first;v=min(adj[first])
 while v!=first:
  if v in ring:return None
  ring.append(v);nxt=adj[v][0] if adj[v][0]!=prev else adj[v][1];prev,v=v,nxt
 return ring if len(ring)==len(old) else None
def best_triangulation(p,ring,d,e):
 n=len(ring);triples=list(combinations(range(n),3));candidate=np.array([tet for a,b,c in triples for tet in [[ring[a],ring[b],ring[c],d],[ring[a],ring[b],ring[c],e]]],np.int64)
 det,q=m.measurements(p,candidate);ds=det.reshape(-1,2);qs=np.abs(q.reshape(-1,2));valid=np.all(np.isfinite(ds),axis=1)&(ds[:,0]*ds[:,1]<0)
 bad=((np.abs(ds)/6<=1e-18)|(qs<1e-5)).sum(axis=1)
 score=np.minimum(np.abs(ds)/6/1e-18,qs/1e-5).min(axis=1)
 tri={key:(int(b),float(sc)) for key,b,sc,ok in zip(triples,bad,score,valid) if ok}
 dp={};split={}
 for i in range(n-1):dp[i,i+1]=(0,float('inf'))
 for width in range(2,n):
  for i in range(n-width):
   j=i+width;best=None
   for k in range(i+1,j):
    if (i,k,j) not in tri or (i,k) not in dp or (k,j) not in dp:continue
    b,sc=tri[i,k,j];left=dp[i,k];right=dp[k,j];value=(b+left[0]+right[0],min(sc,left[1],right[1]))
    if best is None or (value[0],-value[1],k)<(best[0],-best[1],split[i,j]):best=value;split[i,j]=k
   if best is not None:dp[i,j]=best
 if (0,n-1) not in dp:return None
 stack=[(0,n-1)];tris=[]
 while stack:
  i,j=stack.pop()
  if j-i<2:continue
  k=split[i,j];tris.append((ring[i],ring[k],ring[j]));stack.extend([(i,k),(k,j)])
 new=m.orient(p,[[a,b,c,apex] for a,b,c in tris for apex in (d,e)])
 return new,dp[0,n-1],tris
def fixture():
 p=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,-.9999999,1],[0,-.9999999,-1]],np.float32)
 old=m.orient(p,[[4,5,i,(i+1)%4] for i in range(4)])
 ring=ring_order(old,4,5);assert ring is not None
 new,score,tris=best_triangulation(p,ring,4,5)
 assert sum(~m.passes(*m.measurements(p,old)))>sum(~m.passes(*m.measurements(p,new)))
 assert m.certify(p,old,new)==(True,'PASS')
 assert ring_order(old[:3],4,5) is None
 assert m.certify(p,old,new[:-1])[0] is False
 from tempfile import TemporaryDirectory
 from types import SimpleNamespace
 with TemporaryDirectory(prefix='numi-polygon-flip-control-') as td:
  td=Path(td);np.savez_compressed(td/'input.npz',points_world_m=p,tetrahedra=old,material_indices=np.full(len(old),15,np.int32))
  run(SimpleNamespace(input_npz=td/'input.npz',output=td/'output',seconds=30,maximum_degree=4,maximum_operations=20))
  rr=json.loads((td/'output/report.json').read_text());assert rr['passing_operations']==1 and rr['bad_cells_after']==0
 return {'status':'PASS_FIXED_BOUNDARY_POLYGON_EDGE_FLIP_FIXTURES','controls':['closed_ring','reject_open_boundary_edge','dynamic_program_quality','positive_exact_cavity_volume','oriented_boundary','incomplete_replacement_reject'],'physical_admission':False,'native_steps':0}
def run(config):
 start=time.monotonic();config.output.mkdir(parents=True,exist_ok=False)
 with np.load(config.input_npz) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
 det,q=m.measurements(p,t);good=m.passes(det,q);assert np.all(det>0)
 pattern=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
 ed=np.sort(t[:,pattern].reshape(-1,2),axis=1);order=np.lexsort((ed[:,1],ed[:,0]));se=ed[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(se[1:]!=se[:-1],axis=1))];count=np.diff(np.r_[starts,len(se)])
 # Stable degree then edge-key order. Each candidate is a complete current edge star.
 candidates=[]
 for degree in range(4,config.maximum_degree+1):
  ss=starts[count==degree];rr=order[ss[:,None]+np.arange(degree)]//6
  valid=np.any(~good[rr],axis=1)&np.all(mat[rr]==mat[rr[:,0,None]],axis=1)
  candidates.extend((tuple(map(int,edge)),rows.copy()) for edge,rows in zip(se[ss[valid]],rr[valid]))
 del ed,order,se,starts,count
 flat=t.ravel();corner=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 def edge_cells(a,b):return np.intersect1d(corner[offset[a]:offset[a+1]]//4,corner[offset[b]:offset[b+1]]//4,assume_unique=True)
 used=set();removed=[];newall=[];newmat=[];parentrows=[];parentoffset=[0];newoffset=[0];ops=[];rej=Counter()
 for (d,e),rows in candidates:
  if time.monotonic()-start>config.seconds or len(ops)>=config.maximum_operations:break
  old=t[rows];vertices=set(map(int,old.ravel()))
  if vertices&used:rej['conflict']+=1;continue
  ring=ring_order(old,d,e)
  if ring is None:rej['open_or_non_simple_edge_link']+=1;continue
  proposal=best_triangulation(p,ring,d,e)
  if proposal is None:rej['no_positive_polygon_triangulation']+=1;continue
  new,score,tris=proposal
  if score[0]>=sum(~good[rows]):rej['quality_no_progress']+=1;continue
  perimeter={tuple(sorted((ring[i],ring[(i+1)%len(ring)]))) for i in range(len(ring))}
  diagonals={tuple(sorted(edge)) for tri in tris for edge in combinations(tri,2)}-perimeter
  if any(len(edge_cells(a,b)) for a,b in diagonals):rej['new_diagonal_exists_outside']+=1;continue
  # Ring triangles separating the two apex fans must not be pre-existing exterior faces.
  badface=False
  for a,b,c in tris:
   ab=edge_cells(a,b);cc=corner[offset[c]:offset[c+1]]//4
   if len(np.intersect1d(ab,cc,assume_unique=True)):badface=True;break
  if badface:rej['new_separator_face_exists']+=1;continue
  ok,why=m.certify(p,old,new)
  if not ok:rej[why]+=1;continue
  dn,qn=m.measurements(p,new)
  assert np.all(dn>0) and sum(~m.passes(dn,qn))<sum(~good[rows])
  used.update(vertices);removed.extend(rows.tolist());parentrows.extend(rows.tolist());parentoffset.append(len(parentrows));newall.extend(new.tolist());newoffset.append(len(newall));newmat.extend([int(mat[rows[0]])]*len(new))
  ops.append({'old_rows':rows.tolist(),'removed_edge':[d,e],'ring':ring,'new_cell_count':len(new),'before_bad':int(sum(~good[rows])),'after_bad':int(sum(~m.passes(dn,qn)))})
 keep=np.ones(len(t),bool);keep[removed]=False;newall=np.asarray(newall,np.int64).reshape(-1,4)
 output=np.vstack([t[keep],newall]);outputmat=np.r_[mat[keep],np.asarray(newmat,np.int32)];dd,qq=m.measurements(p,output)
 np.savez_compressed(config.output/'fixed-boundary-volume-proposal.npz',points_world_m=p,tetrahedra=output,material_indices=outputmat,retained_input_cell_rows=np.flatnonzero(keep),replacement_parent_offsets=np.asarray(parentoffset,np.int64),replacement_parent_cell_rows=np.asarray(parentrows,np.int64),replacement_tet_offsets=np.asarray(newoffset,np.int64),replacement_tetrahedra=newall)
 r={'status':'PASS_FIXED_BOUNDARY_POLYGON_EDGE_FLIP_PROPOSAL' if ops else 'NO_VALID_CONNECTIVITY_PROGRESS','physical_admission':False,'native_steps':0,'input_npz':str(config.input_npz),'input_npz_sha256':sha(config.input_npz),'script_sha256':sha(Path(__file__)),'cavity_predicate_sha256':sha(H/'fixed_boundary_volume_flip_198_root.py'),'nodes_bit_identical':True,'oriented_boundary_exactly_unchanged':True,'exact_F32_volume_unchanged':True,'material_preserved':True,'eligible_stars':len(candidates),'tested_stars':len(ops)+sum(rej.values()),'passing_operations':len(ops),'rejections':dict(rej),'input_cells':len(t),'output_cells':len(output),'bad_cells_before':int(sum(~good)),'bad_cells_after':int(sum(~m.passes(dd,qq))),'minimum_normalized_det_before':float(q.min()),'minimum_normalized_det_after':float(qq.min()),'minimum_volume_before_m3':float(det.min()/6),'minimum_volume_after_m3':float(dd.min()/6),'elapsed_s':time.monotonic()-start,'operations':ops}
 (config.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='operations'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--input-npz',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--maximum-degree',type=int,default=6);ap.add_argument('--maximum-operations',type=int,default=10000);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture()))
 else:run(a)
