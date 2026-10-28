#!/usr/bin/env python3
"""Kind-aware D37 ABI5 source preparation; separates legacy coarse IDs, binding support, and geometry registration. Never physical admission."""
import argparse,csv,gzip,hashlib,heapq,json,math,struct,time
from pathlib import Path
import numpy as np
D37="7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb"
SOURCE_CSV_SHA="5f90732814c2eb7d372c25ff1336e5e884df28d4463185bc70e398aecfef0e61"
POSE_SHA="0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd"
IDENTITY_SHA="28059226506d5602cdae1c8af2627c7a8bfa18f4b010e6c047affee29c6671e0"
P1_SHA="21f1869b418a2eca1e5a0522059c8f9cc6f8afe1a270ccc7c7d99c9c19fdd2a4"
CONDITIONED_CAND="5e686b14d9360267f4b40bbeb5ed6c9c68bf871933874b5aa8eaa5898757ec2b"
COMPACT_MAP_SHA="5ed0c63d4a35a023bd91d0cbec1a933c8ad1f7b5fc1d32148cd8568ac77db8f4"
CAND="549fab40b3cdfec2d0e1c9a004451d276f87b445f1c7dc9d5a56c1b5a1a53be4"
BASE="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
NB=86; NV=61950; FP=835155552

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def qmat(q):
 x,y,z,w=np.asarray(q,dtype=np.float64)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],np.float64)

def rot(q,p):
 q=np.asarray(q,np.float32);p=np.asarray(p,np.float32)
 x,y,z=q[:3];w=q[3];px,py,pz=p[...,0],p[...,1],p[...,2]
 u=np.float32(2)*(y*pz-z*py);v=np.float32(2)*(z*px-x*pz);t=np.float32(2)*(x*py-y*px)
 return np.stack((px+w*u+y*t-z*v,py+w*v+z*u-x*t,pz+w*t+x*v-y*u),axis=-1).astype(np.float32)

def load_skin(p):
 r=Path(p).read_bytes()
 magic,abi,nb,nv,ni,fp,source=struct.unpack_from("<8s5I32s",r)
 if (magic,abi,nb,nv,ni,fp)!=(b"NHSKIN1\0",5,NB,NV,326961,FP):raise ValueError("D37 NHSKIN ABI/count/fingerprint mismatch")
 bo=60;vo=bo+36*nb;fo=vo+56*nv;wo=fo+4*ni
 if len(r)!=wo+nv*nb*4:raise ValueError("D37 ABI5 byte count mismatch")
 binds=[]
 for i in range(nb):
  x=struct.unpack_from("<I8f",r,bo+36*i)
  binds.append({"body":int(x[0]),"tr":np.array(x[1:4],np.float32),"q":np.array(x[4:8],np.float32),"s":np.float32(x[8])})
 verts=np.empty((nv,6),np.float32)
 for i in range(nv):verts[i]=struct.unpack_from("<6f",r,vo+56*i)
 faces=np.frombuffer(r,"<u4",nv*0+ni,fo).reshape(-1,3).copy()
 weights=np.frombuffer(r,"<f4",nv*nb,wo).reshape(nv,nb).copy()
 return r,source,binds,verts,faces,weights,r[bo:vo]

def load_pose(p,binds):
 x=json.loads(Path(p).read_text())
 if (x.get("schema"),x.get("registration_fingerprint32"),x.get("binding_count"),x.get("payload_abi"),x.get("vertex_count"))!=("numi.human.native-skin-pose-snapshot.v1",FP,NB,5,NV):raise ValueError("pose snapshot does not match D37 ABI5")
 b={}
 for y in x["bodies"]:
  i=int(y["body_index"]);b[i]={}
  for s in ("current","rest"):
   b[i][s+"_p"]=np.asarray(y[s]["position_world_m"],np.float32);b[i][s+"_q"]=np.asarray(y[s]["orientation_world_xyzw"],np.float32)
 if set(z["body"] for z in binds)-set(b):raise ValueError("D37 matched poses omit a source binding body")
 return x,b

def replay(local,w,binds,bodies,state):
 out=np.zeros((len(local),3),np.float32)
 for j,b in enumerate(binds):
  p=bodies[b["body"]]
  q=p[state+"_q"];origin=p[state+"_p"]
  x=b["tr"]+rot(b["q"],local)*b["s"]
  world=origin+rot(q,x)
  out+=w[:,j,None]*world
 return out

def read_source_csv(p):
 rows=[]
 with gzip.open(p,"rt",newline="") as f:
  for r in csv.DictReader(f):
   if int(r["vertex"])!=len(rows):raise ValueError("D37 CSV row identity gap")
   rows.append(r)
 if len(rows)!=NV:raise ValueError("D37 CSV row count mismatch")
 xyz=np.asarray([[float(r[k]) for k in ("wx","wy","wz")] for r in rows],np.float32)
 return rows,xyz

def build_source_crosswalk(p1_path,csvrows):
 # This crosswalk separates legacy baseline node IDs from detail-source rows.
 # For kind 0, node_source_row is a coarse representative row; P1 status-1
 # representatives provide binding support only, never topology ancestry.
 if sha(p1_path)!=P1_SHA:raise ValueError("P1 coarse/detail crosswalk SHA differs from pinned run-006 artifact")
 p1=np.load(p1_path,allow_pickle=False)
 required=("detail_row","detail_source_vertex_id","detail_donor_body","representative_coarse_row","source_status")
 if any(k not in p1.files for k in required):raise ValueError("P1 crosswalk schema missing required arrays")
 detail_row=np.asarray(p1["detail_row"],np.int64);detail_id=np.asarray(p1["detail_source_vertex_id"],np.int64)
 detail_donor=np.asarray(p1["detail_donor_body"],np.int64);rep=np.asarray(p1["representative_coarse_row"],np.int64)
 status=np.asarray(p1["source_status"],np.int64)
 if len(detail_id)!=NV or len(detail_row)!=NV or not np.array_equal(detail_row,np.arange(NV)):
  raise ValueError("P1 detail row domain differs from pinned D37 ABI5 vertex count")
 if len(np.unique(detail_id))!=NV or not np.array_equal(np.sort(detail_id),np.arange(NV)):
  raise ValueError("P1 detail source IDs are not a bijection to pinned D37 vertex IDs")
 if not (len(detail_donor)==len(rep)==len(status)==NV):raise ValueError("P1 crosswalk arrays have inconsistent lengths")
 csv_vertex=np.asarray([int(r["vertex"]) for r in csvrows],np.int64)
 csv_donor=np.asarray([int(r["donor"]) for r in csvrows],np.int64)
 if not np.array_equal(csv_vertex,np.arange(NV)):raise ValueError("current D37 CSV vertex rows are not the pinned source payload row domain")
 id_to_p1={int(v):int(i) for i,v in enumerate(detail_id)}
 id_to_payload={int(csv_vertex[i]):int(i) for i in range(NV)}
 if set(id_to_p1)!=set(id_to_payload):raise ValueError("P1 and current D37 payload source vertex ID domains differ")
 for source_id,p1row in id_to_p1.items():
  payload_row=id_to_payload[source_id]
  if int(detail_donor[p1row])!=int(csv_donor[payload_row]):raise ValueError("P1 detail ID to current D37 payload donor mapping mismatch")
 valid=(status==1)&(rep>=0)
 vals,cnts=np.unique(rep[valid],return_counts=True)
 rep_to_p1={int(r):int(np.flatnonzero(valid&(rep==r))[0]) for r,n in zip(vals,cnts) if n==1}
 return {"p1":p1,"detail_id":detail_id,"detail_donor":detail_donor,"rep":rep,"status":status,
         "id_to_p1":id_to_p1,"id_to_payload":id_to_payload,"rep_to_p1":rep_to_p1,
         "csv_vertex":csv_vertex,"csv_donor":csv_donor}

def load_compact_candidate_rows(path,canonical_vertex_count,candidate_to_o033):
 expected=("compact_vertex_row","canonical_candidate_vertex_row","O033_source_vertex_row")
 compact=[];candidate=[];o033=[]
 with open(path,newline="") as f:
  reader=csv.DictReader(f)
  if any(k not in (reader.fieldnames or ()) for k in expected):raise ValueError("compact O node map header differs from pinned row contract")
  for r in reader:
   compact.append(int(r["compact_vertex_row"]));candidate.append(int(r["canonical_candidate_vertex_row"]));o033.append(int(r["O033_source_vertex_row"]))
 compact=np.asarray(compact,np.int64);candidate=np.asarray(candidate,np.int64);o033=np.asarray(o033,np.int64)
 if not len(compact) or not np.array_equal(compact,np.arange(len(compact))):raise ValueError("compact O map rows are not contiguous from zero")
 if len(np.unique(candidate))!=len(candidate) or np.any(candidate<0) or np.any(candidate>=canonical_vertex_count):raise ValueError("compact O map candidate rows are not unique in-range rows")
 if np.any(o033!=candidate_to_o033[candidate]):raise ValueError("compact O map O033 rows disagree with canonical candidate source map")
 return candidate,o033

def resolve_baseline_source_rows(node_kind,node_source_row,crosswalk):
 kind=np.asarray(node_kind,np.int64);legacy=np.asarray(node_source_row,np.int64)
 if kind.shape!=legacy.shape:raise ValueError("baseline source kind/row arrays differ in shape")
 payload=np.full(len(kind),-1,np.int64);p1rows=np.full(len(kind),-1,np.int64)
 map_kind=np.full(len(kind),"unmapped",dtype="U80")
 domain=np.full(len(kind),"unmapped_baseline_node",dtype="U64")
 for i,(k,s) in enumerate(zip(kind,legacy)):
  if k in (1,2):
   domain[i]="exact_detail_source_vertex_id"
   if s<0 or int(s) not in crosswalk["id_to_payload"]:raise ValueError(f"kind {k} baseline row {i} lacks an exact P1/current D37 detail ID")
   p1rows[i]=crosswalk["id_to_p1"][int(s)];payload[i]=crosswalk["id_to_payload"][int(s)]
   map_kind[i]="baseline_kind1_2_exact_detail_id_weight_support"
  elif k==0:
   domain[i]="coarse_representative_row"
   if s>=0 and int(s) in crosswalk["rep_to_p1"]:
    pr=crosswalk["rep_to_p1"][int(s)];sid=int(crosswalk["detail_id"][pr])
    p1rows[i]=pr;payload[i]=crosswalk["id_to_payload"][sid]
    map_kind[i]="baseline_kind0_unique_P1_representative_binding_support"
   else:
    map_kind[i]="baseline_kind0_unresolved_coarse_representative"
  elif k==5:
   domain[i]="unmatched_root_node"
   map_kind[i]="baseline_kind5_unmatched_root"
  elif k==-1:
   map_kind[i]="baseline_kind_minus1_unmapped"
  else:
   domain[i]="other_baseline_node_kind"
   map_kind[i]=f"baseline_kind{k}_unmapped"
 return payload,p1rows,map_kind,domain

def tri_closest(p,t):
 a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a;ap=p-a
 dot=lambda x,y:np.einsum("ij,ij->i",x,y)
 d1=dot(ab,ap);d2=dot(ac,ap);N=len(t);z=np.zeros((N,3));todo=np.ones(N,bool)
 m=(d1<=0)&(d2<=0);z[m,0]=1;todo[m]=0
 bp=p-b;d3=dot(ab,bp);d4=dot(ac,bp);m=todo&(d3>=0)&(d4<=d3);z[m,1]=1;todo[m]=0
 vc=d1*d4-d3*d2;m=todo&(vc<=0)&(d1>=0)&(d3<=0);den=d1-d3
 v=np.divide(d1,den,out=np.zeros_like(d1),where=np.abs(den)>1e-30);z[m,0]=1-v[m];z[m,1]=v[m];todo[m]=0
 cp=p-c;d5=dot(ab,cp);d6=dot(ac,cp);m=todo&(d6>=0)&(d5<=d6);z[m,2]=1;todo[m]=0
 vb=d5*d2-d1*d6;m=todo&(vb<=0)&(d2>=0)&(d6<=0);den=d2-d6
 v=np.divide(d2,den,out=np.zeros_like(d2),where=np.abs(den)>1e-30);z[m,0]=1-v[m];z[m,2]=v[m];todo[m]=0
 va=d3*d6-d5*d4;m=todo&(va<=0)&((d4-d3)>=0)&((d5-d6)>=0);den=(d4-d3)+(d5-d6)
 v=np.divide(d4-d3,den,out=np.zeros_like(d4),where=np.abs(den)>1e-30);z[m,1]=1-v[m];z[m,2]=v[m];todo[m]=0
 if todo.any():
  den=va+vb+vc;v=np.divide(vb,den,out=np.zeros_like(vb),where=np.abs(den)>1e-30);w=np.divide(vc,den,out=np.zeros_like(vc),where=np.abs(den)>1e-30)
  z[todo,0]=1-v[todo]-w[todo];z[todo,1]=v[todo];z[todo,2]=w[todo]
 q=np.einsum("ij,ijk->ik",z,t);d=dot(q-p,q-p)
 return d,z,q

class BVH:
 def __init__(self,p,f):
  tri=p[f].astype(np.float64);ok=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>1e-15
  self.src=np.flatnonzero(ok);self.tri=tri[ok];self.cent=self.tri.mean(1);self.order=np.arange(len(self.tri));self.nodes=[]
  def build(a,b):
   n=len(self.nodes);self.nodes.append(None);ii=self.order[a:b];lo=self.tri[ii].min((0,1));hi=self.tri[ii].max((0,1))
   if b-a<=12:self.nodes[n]=(lo,hi,-1,-1,a,b)
   else:
    axis=int(np.argmax(np.ptp(self.cent[ii],axis=0)));mid=(a+b)//2
    self.order[a:b]=ii[np.argpartition(self.cent[ii,axis],mid-a)]
    l=build(a,mid);r=build(mid,b);self.nodes[n]=(lo,hi,l,r,-1,-1)
   return n
  build(0,len(self.order))
 def query(self,p):
  p=np.asarray(p,np.float64);heap=[(0.,0)];best=float("inf");row=-1
  while heap:
   d,n=heapq.heappop(heap)
   if d>best+1e-24:break
   lo,hi,l,r,a,b=self.nodes[n]
   if l<0:
    ids=self.order[a:b];ds,be,qq=tri_closest(p,self.tri[ids]);m=float(ds.min())
    if m<=best+1e-24:
     ties=np.flatnonzero(ds<=m+1e-24);k=min(ties,key=lambda j:int(self.src[ids[j]]))
     rr=int(self.src[ids[k]])
     if m<best-1e-24 or row<0 or rr<row:best=m;row=rr;beta=be[k].copy();proj=qq[k].copy()
   else:
    for ch in (l,r):
     a0,b0=self.nodes[ch][:2];v=np.maximum(np.maximum(a0-p,0),p-b0);v=float(v@v)
     if v<=best+1e-24:heapq.heappush(heap,(v,ch))
  if row<0:raise ValueError("nearest source triangle not found")
  return row,beta,proj,math.sqrt(best)

def f32ulp(a,b):
 x=np.asarray(a,np.float32).view(np.int32).astype(np.int64);y=np.asarray(b,np.float32).view(np.int32).astype(np.int64)
 return np.abs(np.where(x<0,0x80000000-x,x)-np.where(y<0,0x80000000-y,y))

def normals(points,faces):
 tri=points[faces].astype(np.float64);fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 if np.any(np.linalg.norm(fn,axis=1)<=1e-15):raise ValueError("candidate has degenerate faces")
 out=np.zeros_like(points,dtype=np.float64)
 for j in range(3):np.add.at(out,faces[:,j],fn)
 n=np.linalg.norm(out,axis=1)
 if np.any(n<=1e-12):raise ValueError("candidate has isolated/degenerate vertex normal")
 return out/n[:,None]

def capped_signed_volume(points,faces):
 ec=__import__("collections").Counter();directed=[]
 for a,b,c in np.asarray(faces,np.int64):
  for u,v in ((a,b),(b,c),(c,a)):
   ec[tuple(sorted((int(u),int(v))))]+=1;directed.append((int(u),int(v)))
 bd=[(u,v) for u,v in directed if ec[tuple(sorted((u,v))) ]==1]
 nxt={}
 for a,b in bd:
  if a in nxt:raise ValueError("candidate O boundary is not directed one-out")
  nxt[a]=b
 if len(nxt)!=len(bd) or any(v not in nxt for v in nxt.values()):raise ValueError("candidate O boundary is not a closed directed loop set")
 todo=set(nxt);loops=[]
 while todo:
  start=min(todo);cur=start;L=[]
  for _ in range(len(nxt)+1):
   if cur in L:raise ValueError("candidate O boundary repeats before closure")
   L.append(cur);todo.discard(cur);cur=nxt[cur]
   if cur==start:break
  if cur!=start:raise ValueError("candidate O boundary loop failed")
  loops.append(L)
 P=np.asarray(points,np.float64);centers=[P[L].mean(0) for L in loops];Q=np.vstack([P,np.asarray(centers)])
 F=list(np.asarray(faces,np.int64))
 for j,L in enumerate(loops):
  center=len(P)+j
  F.extend([[b,a,center] for a,b in zip(L,L[1:]+L[:1])])
 F=np.asarray(F,np.int64)
 vol=float(np.einsum("ij,ij->",Q[F[:,0]],np.cross(Q[F[:,1]],Q[F[:,2]]))/6.0)
 return vol,sorted(map(len,loops))

def csvout(p,head,rows,gz=False):
 op=gzip.open if gz else open
 with op(p,"wt",newline="") as f:
  w=csv.writer(f);w.writerow(head);w.writerows(rows)

def main():
 a=argparse.ArgumentParser()
 for x in ("source-payload","source-csv-gzip","source-poses","candidate","baseline","identity","p1-crosswalk","output-dir"):a.add_argument("--"+x,required=True)
 a.add_argument("--conditioned-candidate",default=None)
 a.add_argument("--compact-node-map",default=None)
 a.add_argument("--max-projection-distance-m",type=float,default=None);a=a.parse_args()
 start=time.time();paths={k:Path(getattr(a,k.replace("-","_"))) for k in ("source-payload","source-csv-gzip","source-poses","candidate","baseline","identity","p1-crosswalk")}
 if a.conditioned_candidate:paths["conditioned-candidate"]=Path(a.conditioned_candidate)
 if a.compact_node_map:paths["compact-node-map"]=Path(a.compact_node_map)
 if bool(a.conditioned_candidate)!=bool(a.compact_node_map):raise ValueError("conditioned O requires its explicit compact-node-map CSV")
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 hashes={k:sha(v) for k,v in paths.items()}
 if hashes["source-payload"]!=D37 or hashes["source-csv-gzip"]!=SOURCE_CSV_SHA or hashes["source-poses"]!=POSE_SHA or hashes["candidate"]!=CAND or hashes["baseline"]!=BASE or hashes["identity"]!=IDENTITY_SHA or hashes["p1-crosswalk"]!=P1_SHA:raise ValueError("input SHA differs from pinned D37/candidate033/baseline806/identity/P1 crosswalk")
 if a.conditioned_candidate and hashes["conditioned-candidate"]!=CONDITIONED_CAND:raise ValueError("conditioned candidate SHA differs from pinned canonical conditioned O")
 if a.compact_node_map and hashes["compact-node-map"]!=COMPACT_MAP_SHA:raise ValueError("compact node map SHA differs from pinned canonical O compact map")
 raw,source_sha,binds,sv,sf,sw,binding_bytes=load_skin(paths["source-payload"])
 pose,bodies=load_pose(paths["source-poses"],binds);csvrows,srcworld=read_source_csv(paths["source-csv-gzip"])
 source_replay=replay(sv[:,:3],sw,binds,bodies,"current");source_err=np.linalg.norm(source_replay.astype(float)-srcworld.astype(float),axis=1)
 if source_err.max()>2e-6:raise ValueError("D37 payload/pose/CSV F32 replay mismatch >2um")
 crosswalk=build_source_crosswalk(paths["p1-crosswalk"],csvrows)
 source_ca=np.load(paths["candidate"],allow_pickle=False)
 canonical_candidate_rows=None
 if a.conditioned_candidate:
  ca=np.load(paths["conditioned-candidate"],allow_pickle=False)
  canonical_points=np.asarray(ca["points_world_m"],np.float32);canonical_faces=np.asarray(ca["source_O_faces"],np.int64)
  if canonical_points.shape!=np.asarray(source_ca["points_world_m"]).shape or not np.array_equal(canonical_points,source_ca["points_world_m"]):raise ValueError("conditioned O must preserve the exact O033 Float32 point table")
  if "candidate_vertex_to_O033_row" not in ca.files:raise ValueError("conditioned O is missing explicit candidate-to-O033 vertex map")
  candidate_to_o033=np.asarray(ca["candidate_vertex_to_O033_row"],np.int64)
  if candidate_to_o033.shape!=(len(canonical_points),) or np.any(candidate_to_o033<0) or np.any(candidate_to_o033>=len(source_ca["points_world_m"])) or len(np.unique(candidate_to_o033))!=len(canonical_points):raise ValueError("conditioned candidate-to-O033 vertex map is incomplete or non-bijective")
  if not np.array_equal(canonical_points,source_ca["points_world_m"][candidate_to_o033]):raise ValueError("conditioned O candidate-to-O033 map disagrees with exact point coordinates")
  canonical_candidate_rows,_=load_compact_candidate_rows(paths["compact-node-map"],len(canonical_points),candidate_to_o033)
  if "output_face_to_precontraction_candidate_face_row" not in ca.files:raise ValueError("conditioned O is missing face-to-O033 source map")
  face_parent=np.asarray(ca["output_face_to_precontraction_candidate_face_row"],np.int64)
  if face_parent.shape!=(len(canonical_faces),) or np.any(face_parent<0) or np.any(face_parent>=len(source_ca["source_O_faces"])):raise ValueError("conditioned face-to-O033 map is incomplete or outside source face range")
  required_ancestry=("face_ancestry_offsets","face_ancestry_parent_face_rows","face_ancestry_source_original_ids","face_ancestry_source_face_rows")
  if any(k not in ca.files for k in required_ancestry):raise ValueError("conditioned O lacks complete per-face ancestry arrays")
  face_ancestry_offsets=np.asarray(ca["face_ancestry_offsets"],np.int64);face_ancestry_parents=np.asarray(ca["face_ancestry_parent_face_rows"],np.int64);face_ancestry_ids=np.asarray(ca["face_ancestry_source_original_ids"],np.int64);face_ancestry_sources=np.asarray(ca["face_ancestry_source_face_rows"],np.int64)
  if face_ancestry_offsets.shape!=(len(canonical_faces)+1,) or face_ancestry_offsets[0]!=0 or face_ancestry_offsets[-1]!=len(face_ancestry_parents) or not np.array_equal(np.diff(face_ancestry_offsets)>0,np.ones(len(canonical_faces),bool)) or len(face_ancestry_ids)!=len(face_ancestry_parents) or len(face_ancestry_sources)!=len(face_ancestry_parents):raise ValueError("conditioned O per-face ancestry CSR is malformed or incomplete")
  compact_inverse=np.full(len(canonical_points),-1,np.int64);compact_inverse[canonical_candidate_rows]=np.arange(len(canonical_candidate_rows),dtype=np.int64)
  if np.any(compact_inverse[canonical_faces]<0):raise ValueError("compact O map omits a vertex used by a conditioned face")
  points=canonical_points[canonical_candidate_rows];faces=compact_inverse[canonical_faces]
  candidate_to_o033=candidate_to_o033[canonical_candidate_rows]
  if not np.array_equal(np.unique(faces),np.arange(len(points),dtype=np.int64)):raise ValueError("compact O faces leave an unused compact vertex or missing compact row")
  face_ancestry_offsets=face_ancestry_offsets
 else:
  ca=source_ca;points=np.asarray(ca["points_world_m"],np.float32);faces=np.asarray(ca["source_O_faces"],np.int64)
  candidate_to_o033=np.arange(len(points),dtype=np.int64);canonical_candidate_rows=np.arange(len(points),dtype=np.int64);face_parent=np.arange(len(faces),dtype=np.int64)
  face_ancestry_offsets=np.arange(len(faces)+1,dtype=np.int64);face_ancestry_parents=np.arange(len(faces),dtype=np.int64);face_ancestry_ids=np.ones(len(faces),np.int64);face_ancestry_sources=np.arange(len(faces),dtype=np.int64)
 if points.shape!=((32002 if a.conditioned_candidate else 32218),3) or faces.ndim!=2 or faces.shape[1]!=3 or len(faces)!=(63959 if a.conditioned_candidate else 64391):raise ValueError("O033/conditioned compact O dimensions mismatch")
 im=np.load(paths["identity"],allow_pickle=False);o033_base=np.asarray(im["output_vertex_source_rows"],np.int64);o033_facebase=np.asarray(im["output_face_exact_source_rows"],np.int64)
 if o033_base.shape!=(len(source_ca["points_world_m"]),) or o033_facebase.shape!=(len(source_ca["source_O_faces"]),):raise ValueError("O033 identity map shape mismatch")
 outbase=o033_base[candidate_to_o033]
 if a.conditioned_candidate:
  canonical_points_for_face_map=np.asarray(ca["points_world_m"],np.float32)
  canonical_faces_for_face_map=np.asarray(ca["source_O_faces"],np.int64)
  canonical_to_o033=np.asarray(ca["candidate_vertex_to_O033_row"],np.int64)
  mapped_triangles=canonical_to_o033[canonical_faces_for_face_map]
  source_triangles=np.asarray(source_ca["source_O_faces"],np.int64)[face_parent]
  same_triangle=np.all(np.sort(mapped_triangles,axis=1)==np.sort(source_triangles,axis=1),axis=1)
  facebase=np.full(len(faces),-1,np.int64);facebase[same_triangle]=o033_facebase[face_parent[same_triangle]]
 else:
  facebase=o033_facebase.copy()
 base=np.load(paths["baseline"],allow_pickle=False);basesrc=np.asarray(base["node_source_row"],np.int64);basekind=np.asarray(base["node_kind"],np.int64)
 if outbase.shape!=(len(points),) or facebase.shape!=(len(faces),) or len(basesrc)!=21212 or basekind.shape!=basesrc.shape:raise ValueError("806 identity map shape mismatch")
 ok=(outbase>=0)&(outbase<len(basesrc));legacy_kind=np.full(len(points),-1,np.int64);legacy_source=np.full(len(points),-1,np.int64)
 legacy_kind[ok]=basekind[outbase[ok]];legacy_source[ok]=basesrc[outbase[ok]]
 frame_row,p1row,mapping_kind,legacy_domain=resolve_baseline_source_rows(legacy_kind,legacy_source,crosswalk)
 legacy_kind[~ok]=-1;legacy_source[~ok]=-1;legacy_domain[~ok]="no_baseline806_node_identity";mapping_kind[~ok]="no_baseline806_identity_geometry_registration"
 mapped_support=frame_row>=0
 exact_detail=np.isin(legacy_kind,(1,2))&mapped_support
 # Kind-0 coarse rows use the unique P1 representative only for frame weights.
 # Their O-to-source-surface registration is separately inferred from current triangles.
 needs_projection=~exact_detail
 frow=np.full(len(points),-1,np.int64);beta=np.zeros((len(points),3));proj=np.full((len(points),3),np.nan);dist=np.zeros(len(points))
 if needs_projection.any():
  bvh=BVH(srcworld,sf)
  for i in np.flatnonzero(needs_projection):frow[i],beta[i],proj[i],dist[i]=bvh.query(points[i])
 if a.max_projection_distance_m is not None and dist[needs_projection].max(initial=0)>a.max_projection_distance_m:raise ValueError("projection distance exceeds requested maximum")
 w=np.empty((len(points),NB),np.float64);w[mapped_support]=sw[frame_row[mapped_support]]
 for i in np.flatnonzero(~mapped_support):w[i]=beta[i]@sw[sf[frow[i]]].astype(np.float64)
 sums=w.sum(1)
 if np.any(~np.isfinite(w)) or w.min()<-1e-7 or np.any(sums<=1e-12):raise ValueError("invalid transferred full86 weights")
 w[~mapped_support]/=sums[~mapped_support,None]
 wf=w.astype(np.float32);wf[mapped_support]=sw[frame_row[mapped_support]]
 for i in np.flatnonzero(~mapped_support):
  k=int(np.argmax(wf[i]));wf[i,k]=np.float32(wf[i,k]+np.float32(1)-np.sum(wf[i],dtype=np.float32))
 if np.any(wf<0) or np.max(np.abs(wf.sum(1,dtype=np.float64)-1))>2e-6:raise ValueError("output full86 weights do not close")
 A=[];c=[]
 for b in binds:
  p=bodies[b["body"]];R=qmat(p["current_q"]);A.append(R@qmat(b["q"])*float(b["s"]));c.append(p["current_p"].astype(float)+R@b["tr"].astype(float))
 A=np.asarray(A);c=np.asarray(c);J=np.einsum("nb,bjk->njk",wf.astype(float),A);bias=np.einsum("nb,bj->nj",wf.astype(float),c)
 det=np.linalg.det(J);cond=np.linalg.cond(J)
 if np.any(det<=1e-10) or np.any(~np.isfinite(det)):raise ValueError("weighted-LBS inverse has nonpositive/singular Jacobian")
 if np.any(cond>=10) or np.any(~np.isfinite(cond)):raise ValueError("weighted-LBS inverse condition >=10")
 local=np.linalg.solve(J,(points.astype(float)-bias)[...,None])[...,0].astype(np.float32)
 replayed=replay(local,wf,binds,bodies,"current");res=replayed.astype(float)-points.astype(float);resn=np.linalg.norm(res,axis=1);ulp=f32ulp(replayed,points)
 rawnorm=normals(points,faces)
 capped_volume,loop_lengths=capped_signed_volume(points,faces)
 direct_source_normals=np.asarray([[float(csvrows[int(frame_row[i])][k]) for k in ("nx","ny","nz")] for i in np.flatnonzero(exact_detail)],np.float64)
 raw_alignment=np.einsum("ij,ij->i",rawnorm[exact_detail],direct_source_normals)
 if not len(raw_alignment):raise ValueError("no exact detail-ID rows available for source normal orientation check")
 if capped_volume<0 and float(np.median(raw_alignment))<0:normal_sign=-1
 elif capped_volume>0 and float(np.median(raw_alignment))>0:normal_sign=1
 else:raise ValueError("temporary-capped face orientation and direct D37 normal alignment disagree")
 currnorm=(normal_sign*rawnorm).astype(np.float64)
 outward_alignment=normal_sign*raw_alignment
 if float(np.median(outward_alignment))<=0:raise ValueError("outward-normal policy failed direct D37 alignment")
 nb=[]
 for b in binds:
  p=bodies[b["body"]];nb.append(qmat(p["current_q"])@qmat(p["rest_q"]).T)
 nb=np.asarray(nb);N=np.einsum("nb,bjk->njk",wf.astype(float),nb);ndet=np.linalg.det(N);ncond=np.linalg.cond(N)
 if np.any(ndet<=1e-10) or np.any(~np.isfinite(ndet)) or np.any(ncond>=1e6) or np.any(~np.isfinite(ncond)):raise ValueError("world-rest normal inverse singular/reflected/ill-conditioned")
 restnorm=np.linalg.solve(N,currnorm[...,None])[...,0];restnorm/=np.linalg.norm(restnorm,axis=1)[:,None];restnorm=restnorm.astype(np.float32)
 # Reproduce ABI5 world-rest normal transport in Float32 operation order.
 nacc=np.zeros((len(points),3),np.float32)
 for j,b in enumerate(binds):
  po=bodies[b["body"]];q=po["rest_q"].copy();q[:3]*=-1
  nacc+=wf[:,j,None]*rot(po["current_q"],rot(q,restnorm))
 nlen=np.linalg.norm(nacc,axis=1).astype(np.float32)
 if np.any(nlen<=1e-6):raise ValueError("native normal replay collapsed at a candidate vertex")
 nreplay=(nacc/nlen[:,None]).astype(np.float32);nres=nreplay.astype(float)-currnorm;nresn=np.linalg.norm(nres,axis=1);nulp=f32ulp(nreplay,currnorm.astype(np.float32))
 if np.any(np.einsum("ni,ni->n",nreplay,currnorm)<=0):raise ValueError("native normal replay reversed a candidate normal")
 # Current-reference Float32 geometry audit, separate from source-rest pose flips.
 rc=points[faces].astype(float);rrc=replayed[faces].astype(float)
 rcross=np.cross(rc[:,1]-rc[:,0],rc[:,2]-rc[:,0]);rrcross=np.cross(rrc[:,1]-rrc[:,0],rrc[:,2]-rrc[:,0])
 rcd=np.einsum("ij,ij->i",rcross,rrcross);rca=np.linalg.norm(rcross,axis=1);rrca=np.linalg.norm(rrcross,axis=1)
 current_flip=(rcd<=0)|(rrca<=1e-15)
 # Replay geometry in source-rest pose and retain orientation audit without concealing flips.
 Ar=[];cr=[]
 for b in binds:
  p=bodies[b["body"]];R=qmat(p["rest_q"]);Ar.append(R@qmat(b["q"])*float(b["s"]));cr.append(p["rest_p"].astype(float)+R@b["tr"].astype(float))
 Jr=np.einsum("nb,bjk->njk",wf.astype(float),np.asarray(Ar));br=np.einsum("nb,bj->nj",wf.astype(float),np.asarray(cr))
 restworld=np.einsum("nij,nj->ni",Jr,local.astype(float))+br
 tc=points[faces].astype(float);tr=restworld[faces];nc=np.cross(tc[:,1]-tc[:,0],tc[:,2]-tc[:,0]);nr=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0])
 fd=np.einsum("ij,ij->i",nc,nr);ac=np.linalg.norm(nc,axis=1);ar=np.linalg.norm(nr,axis=1);flip=(fd<=0)|(ar<=1e-15)
 owners=np.empty(len(points),np.int64)
 for i in range(len(points)):
  owners[i]=int(csvrows[frame_row[i]]["donor"]) if mapped_support[i] else int(binds[int(np.argmax(wf[i]))]["body"])
 bodyids=np.asarray([b["body"] for b in binds],np.int64)
 localdon=np.empty_like(points,dtype=float);mat=np.empty((len(points),9),float)
 for i in range(len(points)):
  q=bodies[int(owners[i])];R=qmat(q["current_q"]);M=R.T;mat[i]=M.reshape(-1);localdon[i]=M@(points[i].astype(float)-q["current_p"].astype(float))
 source_vertex_row=np.where(exact_detail,frame_row,-1)
 witness_output_row=int(np.flatnonzero(canonical_candidate_rows==16398)[0]) if np.any(canonical_candidate_rows==16398) else -1
 if witness_output_row<0:raise ValueError("canonical row 16398 is missing from the emitted O vertex map")
 source_geometry_point=np.empty_like(points,dtype=np.float64)
 source_geometry_point[exact_detail]=srcworld[source_vertex_row[exact_detail]].astype(np.float64)
 source_geometry_point[needs_projection]=proj[needs_projection]
 source_registration_distance=np.linalg.norm(points.astype(np.float64)-source_geometry_point,axis=1)
 source_domain=np.full(len(points),"D37_NHSKIN_triangle_inferred",dtype="U48")
 source_domain[exact_detail]="D37_NHSKIN_detail_vertex_id"
 frame_owner_method=np.full(len(points),"D37_NHSKIN_triangle_barycentric_weight_blend",dtype="U96")
 frame_owner_method[mapped_support]=np.char.add(mapping_kind[mapped_support],"_full86_weight_row")
 source_registration_method=np.full(len(points),"nearest_D37_triangle_geometry_registration",dtype="U64")
 source_registration_method[exact_detail]="exact_D37_detail_source_vertex_id_registration"
 d37_binding_support_id=np.full(len(points),-1,np.int64)
 d37_binding_support_id[mapped_support]=crosswalk["detail_id"][p1row[mapped_support]]
 header=["vertex","canonical_candidate_vertex_row","O033_source_vertex_row","donor","wx","wy","wz","nx","ny","nz","lx","ly","lz"]+[f"m{r}{c}" for r in range(3) for c in range(3)]+["baseline806_node_row","baseline806_node_kind","baseline806_node_source_row","baseline806_source_domain","source_domain","source_mapping_kind","source_registration_method","frame_owner_method","p1_crosswalk_row","d37_source_vertex_row","d37_binding_support_vertex_row","d37_binding_support_vertex_id","d37_source_face_row","beta0","beta1","beta2","projection_distance_m","source_registration_distance_m","residual_x","residual_y","residual_z"]
 regs=[]
 for i in range(len(points)):
  rr=points[i].astype(float)-source_geometry_point[i]
  regs.append([i,int(canonical_candidate_rows[i]),int(candidate_to_o033[i]),int(owners[i]),*map(float,points[i]),*map(float,currnorm[i]),*map(float,localdon[i]),*map(float,mat[i]),int(outbase[i]),int(legacy_kind[i]),int(legacy_source[i]),str(legacy_domain[i]),str(source_domain[i]),str(mapping_kind[i]),str(source_registration_method[i]),str(frame_owner_method[i]),int(p1row[i]),int(source_vertex_row[i]),int(frame_row[i]),int(d37_binding_support_id[i]),int(frow[i]),*map(float,beta[i]),float(dist[i]),float(source_registration_distance[i]),*map(float,rr)])
 reg=out/"registered-O.csv";csvout(reg,header,regs)
 fc=out/"registered-O-faces.csv";csvout(fc,["face","v0","v1","v2","O033_source_face_parent_row","baseline806_source_face_row_exact"],[[i,*map(int,faces[i]),int(face_parent[i]),int(facebase[i])] for i in range(len(faces))])
 face_ancestry_path=out/"registered-O-face-ancestry.csv.gz"
 ancestry_rows=([i,j,int(face_ancestry_parents[j]),int(face_ancestry_ids[j]),int(face_ancestry_sources[j])] for i in range(len(faces)) for j in range(int(face_ancestry_offsets[i]),int(face_ancestry_offsets[i+1]))) if a.conditioned_candidate else iter(())
 csvout(face_ancestry_path,["candidate_face_row","ancestry_entry_row","parent_face_row","source_original_id","source_face_row"],ancestry_rows,True)
 wh=["vertex"]+[f"w{i}" for i in range(NB)];wp=out/"full-weights.csv.gz";csvout(wp,wh,([i,*map(float,wf[i])] for i in range(len(points))),True)
 frames=[]
 for j,b in enumerate(binds):
  p=bodies[b["body"]];R=qmat(p["current_q"]);RR=qmat(p["rest_q"])
  frames.append([j,b["body"],*map(float,p["current_p"]),*map(float,R.ravel()),*map(float,p["rest_p"]),*map(float,RR.ravel())])
 fp=out/"donor-frames.csv";csvout(fp,["binding","donor","current_x","current_y","current_z"]+[f"current_R{r}{c}" for r in range(3) for c in range(3)]+["rest_x","rest_y","rest_z"]+[f"rest_R{r}{c}" for r in range(3) for c in range(3)],frames)
 mp=out/"source-transfer-map.csv";csvout(mp,["vertex","canonical_candidate_vertex_row","O033_source_vertex_row","binding_support_method","source_registration_method","baseline806_node_row","baseline806_node_kind","baseline806_node_source_row","baseline806_source_domain","source_domain","p1_crosswalk_row","d37_source_vertex_row","d37_binding_support_vertex_row","d37_binding_support_vertex_id","d37_source_face_row","beta0","beta1","beta2","source_geometry_x","source_geometry_y","source_geometry_z","target_x","target_y","target_z","projection_distance_m","source_registration_distance_m"],([i,int(canonical_candidate_rows[i]),int(candidate_to_o033[i]),str(mapping_kind[i]),str(source_registration_method[i]),int(outbase[i]),int(legacy_kind[i]),int(legacy_source[i]),str(legacy_domain[i]),str(source_domain[i]),int(p1row[i]),int(source_vertex_row[i]),int(frame_row[i]),int(d37_binding_support_id[i]),int(frow[i]),*map(float,beta[i]),*map(float,source_geometry_point[i]),*map(float,points[i]),float(dist[i]),float(source_registration_distance[i])] for i in range(len(points))))
 rp=out/"reference-f32-replay.csv";csvout(rp,["vertex","target_x","target_y","target_z","replay_x","replay_y","replay_z","residual_x","residual_y","residual_z","residual_norm_m","ulp_x","ulp_y","ulp_z","bitwise_exact","position_jacobian_det","position_jacobian_cond","normal_target_x","normal_target_y","normal_target_z","normal_replay_x","normal_replay_y","normal_replay_z","normal_residual_norm","normal_ulp_x","normal_ulp_y","normal_ulp_z","normal_bitwise_exact","normal_blend_det","normal_blend_cond"],([i,*map(float,points[i]),*map(float,replayed[i]),*map(float,res[i]),float(resn[i]),*map(int,ulp[i]),bool(np.array_equal(replayed[i].view(np.uint32),points[i].view(np.uint32))),float(det[i]),float(cond[i]),*map(float,currnorm[i]),*map(float,nreplay[i]),float(nresn[i]),*map(int,nulp[i]),bool(np.array_equal(nreplay[i].view(np.uint32),currnorm[i].astype(np.float32).view(np.uint32))),float(ndet[i]),float(ncond[i])] for i in range(len(points))))
 # Native ABI5 layout: 60-byte header, original 86 bindings, 56-byte vertices, u32 faces, all 86 f32 weights.
 payload=out/"candidate-d37-033.nhskin";head=struct.pack("<8s5I32s",b"NHSKIN1\0",5,NB,len(points),faces.size,FP,source_sha)
 vbytes=bytearray(len(points)*56);ci=np.empty((len(points),4),np.uint32);cw=np.empty((len(points),4),np.float32)
 for i in range(len(points)):
  order=np.lexsort((np.arange(NB),-wf[i]))[:4];ci[i]=order;cw[i]=wf[i,order];cw[i]/=cw[i].sum(dtype=np.float32);cw[i,0]+=np.float32(1)-cw[i].sum(dtype=np.float32)
  struct.pack_into("<6f4I4f",vbytes,i*56,*local[i],*restnorm[i],*ci[i],*cw[i])
 with payload.open("wb") as f:f.write(head);f.write(binding_bytes);f.write(vbytes);f.write(faces.astype("<u4").tobytes());f.write(wf.astype("<f4",copy=False).tobytes())
 combined=hashlib.sha256("".join(hashes[k] for k in sorted(hashes)).encode()).hexdigest();words=[int.from_bytes(bytes.fromhex(combined)[i:i+8],"little") for i in range(0,32,8)]
 hpp=out/"NumiHumanCandidateSkin033.hpp"
 payload_sha=sha(payload);reg_sha=sha(reg);face_sha=sha(fc);weight_sha=sha(wp);frame_sha=sha(fp)
 hpp.write_text("#pragma once\n#include <array>\n#include <cstdint>\n#include <string_view>\nnamespace numi::human::candidate033 {\ninline constexpr std::uint32_t kPayloadABI=5u,kBindingCount=86u,kOuterVertexCount="+str(len(points))+"u,kOuterFaceCount="+str(len(faces))+"u,kRegistrationFingerprint=835155552u;\ninline constexpr std::array<std::uint64_t,4> kReferenceSourceIdentity={"+",".join(str(x)+"ull" for x in words)+"};\n"+
 "inline constexpr std::string_view kNHSKIN=\"candidate-d37-033.nhskin\",kRegisteredOuter=\"registered-O.csv\",kOuterFaces=\"registered-O-faces.csv\",kFull86Weights=\"full-weights.csv.gz\",kDonorFrames=\"donor-frames.csv\",kSourceMap=\"source-transfer-map.csv\",kReplay=\"reference-f32-replay.csv\",kFaceAncestry=\"registered-O-face-ancestry.csv.gz\";\n"+
 "inline constexpr std::string_view kCandidateSHA256=\""+hashes.get("conditioned-candidate",hashes["candidate"])+"\",kO033SourceCandidateSHA256=\""+hashes["candidate"]+"\",kD37PayloadSHA256=\""+hashes["source-payload"]+"\",kD37PoseSHA256=\""+hashes["source-poses"]+"\",kD37RegisteredCSVGzipSHA256=\""+hashes["source-csv-gzip"]+"\",kBaseline806SHA256=\""+hashes["baseline"]+"\",kP1CrosswalkSHA256=\""+hashes["p1-crosswalk"]+"\",kReferenceIdentitySHA256=\""+combined+"\";\n"+
 "inline constexpr std::string_view kOutputPayloadSHA256=\""+payload_sha+"\",kRegisteredOuterSHA256=\""+reg_sha+"\",kOuterFacesSHA256=\""+face_sha+"\",kFull86WeightsSHA256=\""+weight_sha+"\",kDonorFramesSHA256=\""+frame_sha+"\";\ninline constexpr bool kPhysicalAdmission=false;\n}\n")
 receipt={"classification":"D37-derived ABI5 surface source preparation; reference pose replay only",
 "inputs":{k:{"path":str(v),"sha256":hashes[k]} for k,v in paths.items()},
 "identity":{"abi":5,"bindings":NB,"source_vertices":NV,"source_faces":len(sf),"fingerprint":FP,"geometry_parent":"D37","matching_pose_schema":pose["schema"],"mass_ledger_included":False,"native_physical_admission":False},
 "outputs":{str(p.name):{"path":str(p),"sha256":sha(p)} for p in (payload,reg,fc,wp,fp,mp,rp,hpp)} | {face_ancestry_path.name:{"path":str(face_ancestry_path),"sha256":sha(face_ancestry_path)}},
 "fields":{"registered-O.csv":header,"registered-O-faces.csv":["face","v0","v1","v2","O033_source_face_parent_row","baseline806_source_face_row_exact"],"registered-O-face-ancestry.csv.gz":["candidate_face_row","ancestry_entry_row","parent_face_row","source_original_id","source_face_row"],"full-weights.csv.gz":["vertex","w0..w85"]},
 "kind_aware_mapping":{"p1_crosswalk_sha256":P1_SHA,"p1_crosswalk_path":str(paths["p1-crosswalk"]),"p1_detail_id_to_current_payload_row":"validated bijection by source ID; P1 donor equals current D37 registered CSV donor for all 61,950 rows","kind0_node_source_semantics":"coarse representative row; P1 status-1 unique representative is binding-weight support only, not topology ancestry","kind1_2_node_source_semantics":"exact detail_source_vertex_id","other_kind_semantics":"unmatched for direct mapping; use explicit geometric triangle registration only","baseline_node_kind_counts":{str(int(k)):int(np.sum(basekind==k)) for k in np.unique(basekind)},"conditioned_candidate":{"enabled":bool(a.conditioned_candidate),"compact_node_map_sha256":hashes.get("compact-node-map",""),"compact_output_vertex_count":int(len(points)),"canonical_candidate_vertex_rows_sha256":hashlib.sha256(np.ascontiguousarray(canonical_candidate_rows,dtype="<i8").tobytes()).hexdigest(),"candidate_to_O033_vertex_map_sha256":hashlib.sha256(np.ascontiguousarray(candidate_to_o033,dtype="<i8").tobytes()).hexdigest(),"candidate_face_parent_map_sha256":hashlib.sha256(np.ascontiguousarray(face_parent,dtype="<i8").tobytes()).hexdigest(),"faces_directly_matching_O033_source_triangles":int(np.sum(facebase>=0)),"faces_with_only_full_multi_source_ancestry":int(np.sum(facebase<0)),"full_face_ancestry_entries":int(len(face_ancestry_parents) if a.conditioned_candidate else 0),"source_face_ancestry_status":"pinned conditioned-candidate CSR copied losslessly" if a.conditioned_candidate else "no conditioned face CSR; exact source rows remain in registered-O-faces.csv"},"candidate_mapping_counts":{"kind1_2_exact_detail_id_rows":int(np.sum(exact_detail)),"kind0_p1_representative_binding_support_rows":int(np.sum((legacy_kind==0)&mapped_support)),"kind0_unresolved_rows":int(np.sum((legacy_kind==0)&(~mapped_support))),"geometry_triangle_weight_rows":int(np.sum(~mapped_support)),"geometry_triangle_registration_rows":int(np.sum(needs_projection))},"crosswalk_coverage":{"p1_detail_rows":int(len(crosswalk["detail_id"])),"unique_source_ids":int(len(crosswalk["id_to_p1"])),"unique_status1_coarse_representatives":int(len(crosswalk["rep_to_p1"])),"P1_detail_ids_match_current_payload_vertex_rows":bool(np.array_equal(np.sort(crosswalk["detail_id"]),np.arange(NV))),"P1_detail_donors_match_current_payload_donors":bool(np.array_equal(crosswalk["detail_donor"][np.argsort(crosswalk["detail_id"])],crosswalk["csv_donor"]))},"row_16398_witness":{"output_vertex_row":witness_output_row,"canonical_candidate_vertex_row":int(canonical_candidate_rows[witness_output_row]),"O033_source_vertex_row":int(candidate_to_o033[witness_output_row]),"baseline806_node_row":int(outbase[witness_output_row]),"baseline806_node_kind":int(legacy_kind[witness_output_row]),"legacy_node_source_row":int(legacy_source[witness_output_row]),"legacy_source_domain":str(legacy_domain[witness_output_row]),"p1_crosswalk_row":int(p1row[witness_output_row]),"d37_binding_support_vertex_row":int(frame_row[witness_output_row]),"d37_binding_support_vertex_id":int(d37_binding_support_id[witness_output_row]),"d37_source_vertex_row_exact_geometry_ancestry":int(source_vertex_row[witness_output_row]),"d37_geometric_source_face_row":int(frow[witness_output_row]),"source_domain":str(source_domain[witness_output_row]),"source_mapping_kind":str(mapping_kind[witness_output_row]),"source_registration_method":str(source_registration_method[witness_output_row]),"source_registration_distance_m":float(source_registration_distance[witness_output_row]),"legacy_row_used_as_D37_detail_index":False}},
 "checks":{"D37_source_replay_max_error_m":float(source_err.max()),"direct_source_weight_rows":int(exact_detail.sum()),"representative_binding_support_rows":int(np.sum((legacy_kind==0)&mapped_support)),"barycentric_weight_rows":int((~mapped_support).sum()),"mapped_weights_bitwise_preserved":bool(np.array_equal(wf[mapped_support],sw[frame_row[mapped_support]])),
 "weight_min":float(wf.min()),"weight_max":float(wf.max()),"weight_sum_error_max":float(np.max(np.abs(wf.sum(1,dtype=float)-1))),
 "projection_distance_m":{"min":float(dist[needs_projection].min()),"median":float(np.median(dist[needs_projection])),"p95":float(np.quantile(dist[needs_projection],.95)),"max":float(dist[needs_projection].max())},"source_registration_distance_m":{"p50":float(np.median(source_registration_distance)),"p95":float(np.quantile(source_registration_distance,.95)),"max":float(source_registration_distance.max())},
 "position_jacobian_det_min":float(det.min()),"position_jacobian_cond_max":float(cond.max()),"normal_blend_det_min":float(ndet.min()),"normal_blend_cond_max":float(ncond.max()),
 "candidate_outer_normal_orientation":{"candidate_face_winding":"inward" if normal_sign<0 else "outward","temporary_capped_signed_volume_m3":float(capped_volume),"temporary_cap_loop_lengths":loop_lengths,"normal_sign_applied":int(normal_sign),"direct_D37_native_normal_alignment_raw":{"positive_fraction":float(np.mean(raw_alignment>0)),"median_dot":float(np.median(raw_alignment)),"p05_dot":float(np.quantile(raw_alignment,.05)),"p95_dot":float(np.quantile(raw_alignment,.95))},"direct_D37_native_normal_alignment_outward":{"positive_fraction":float(np.mean(outward_alignment>0)),"median_dot":float(np.median(outward_alignment)),"p05_dot":float(np.quantile(outward_alignment,.05)),"p95_dot":float(np.quantile(outward_alignment,.95))}},
 "reference_f32_replay":{"residual_max_m":float(resn.max()),"residual_p95_m":float(np.quantile(resn,.95)),"ulp_max":int(ulp.max()),"bitwise_exact_vertices":int(np.all(ulp==0,axis=1).sum())},
 "reference_f32_normal_replay":{"residual_max":float(nresn.max()),"residual_p95":float(np.quantile(nresn,.95)),"ulp_max":int(nulp.max()),"bitwise_exact_vertices":int(np.all(nulp==0,axis=1).sum())},
 "current_reference_f32_orientation_audit":{"flipped_or_degenerate_faces":int(current_flip.sum()),"degenerate_faces":int((rrca<=1e-15).sum()),"minimum_face_cosine":float(np.min(rcd/np.maximum(rca*rrca,1e-300)))},
 "source_rest_unposed_orientation_audit":{"flipped_or_degenerate_faces":int(flip.sum()),"degenerate_faces":int((ar<=1e-15).sum()),"minimum_face_cosine":float(np.min(fd/np.maximum(ac*ar,1e-300)))}},
 "admission":{"native_runtime_verified":False,"physical_motion_qualified":False,"mass_ledger_pinned":False,"admitted":False},"elapsed_s":time.time()-start}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(out),"payload_sha256":sha(payload),"exact_detail_id_rows":int(exact_detail.sum()),"kind0_representative_support_rows":int(np.sum((legacy_kind==0)&mapped_support)),"geometry_triangle_weight_rows":int((~mapped_support).sum()),"geometry_registration_rows":int(needs_projection.sum()),"projection_max_m":float(dist[needs_projection].max()),"source_registration_max_m":float(source_registration_distance.max()),"replay_max_m":float(resn.max()),"exact_vertices":int(np.all(ulp==0,axis=1).sum()),"normal_replay_max":float(nresn.max()),"current_reference_flips":int(current_flip.sum()),"rest_flips":int(flip.sum()),"receipt":str(out/"receipt.json")},indent=2))
if __name__=="__main__":main()
