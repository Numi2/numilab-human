#!/usr/bin/env python3
"""Independent exact replay for the D500 indexed379 volume quotient."""
from pathlib import Path
import hashlib,json,time,traceback
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight"
I=H/"fat-comp0-connected-warning-repair-412/run-042-current-54381-normalized-rebind/fat-current-typed-target-rebound-54381-lineage-corrected.npz"
O=R/"volume-transfer-500um-001/run001/fat-D500-short-edge-quotient-volume-proposal.npz"
TR=R/"volume-transfer-500um-001/run001/report.json"
OM=R/"volume-transfer-500um-001/run001/operation-volume-star-map.jsonl"
SI=R/"preflight-output/identity-seed-D.npz"
SO=R/"surface-proposal-500um/candidate-D-short-edge-collapse-raw.npz"
SR=R/"surface-proposal-500um/report.json"
SL=R/"surface-proposal-500um/accepted-operations.jsonl"
P=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/run-038-kernel-fans-297/protection-006/comp0-comp1-protected-stars.npz"
SCRIPT=R/"volume-transfer-500um-001/transfer_D500_indexed379.py"
OUT=R/"volume-transfer-500um-001/independent-replay-001"
PIN={"input":"5aa02086aac4b20e879332cd9d5ba64da636dd8a520de54e7b96b00225c7fe42","output":"35a5545ed4d3bbe40499df0b143456683fb8f0739372e2a53b8a3227e4474b24","transfer":"97a48264fcceb8f99b37d18fe74c99d49f4f5c003260f2f92b16f550f44ef664","opmap":"566395d7eb98dc3e45d329d83b2f262782310e4af8f76a2b86576f28d8dd74f9","surface_input":"88c2c7019d15e04f92ddfd5d1518b43863677131cd098ea1e482aa403fe1f599","surface_output":"c87d1efc7d233ab33f49dda1d49fc96d31756b2c696ee94b3b8f8fa77d3aed72","surface_report":"dac747f7d2fecbd36d6dd04f2f5f1ae4c18914834a819237b3acf49e5ee40173","surface_ledger":"4c4eab7d245c00d9433714861d3df6e86a8ebc3d67d855e5856e90dc70115850","protection":"aaa7b1d01d354b6ad18a885359738b84f267e31cb60a9794665f43f98980365d"}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def need(x,msg):
 if not x:raise ValueError(msg)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def chain(f):
 a=np.asarray(f,np.int64).reshape((-1,3));good=np.all(np.diff(np.sort(a,axis=1),axis=1)!=0,axis=1);a=a[good]
 if not len(a):return np.empty(0,np.int64),np.empty(0,np.int64)
 s=np.sort(a,axis=1);inv=(a[:,0]>a[:,1]).astype(np.int8)+(a[:,0]>a[:,2]).astype(np.int8)+(a[:,1]>a[:,2]).astype(np.int8);sg=(1-2*(inv&1)).astype(np.int64)
 k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];su=np.add.reduceat(sg,st);kk=k[st];keep=su!=0;return kk[keep],su[keep]
def boundary(t):
 F=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64);f=np.asarray(t,np.int64)[:,F].reshape((-1,3));s=np.sort(f,axis=1)
 inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8);sg=(1-2*(inv&1)).astype(np.int8)
 k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];ct=np.diff(np.r_[st,len(k)]);su=np.add.reduceat(sg,st)
 if np.any(ct>2) or np.any((ct==2)&(su!=0)):raise ValueError("bad tet face incidence")
 return f[o[st[ct==1]]]
def closed(f,label):
 a=np.asarray(f,np.int64);e=np.concatenate((a[:,[0,1]],a[:,[1,2]],a[:,[2,0]]));sg=np.where(e[:,0]<e[:,1],1,-1).astype(np.int8);s=np.sort(e,axis=1);k=s[:,0]*(1<<20)+s[:,1];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1]
 if np.any(np.add.reduceat(sg,st)):raise ValueError(label+" not closed")
def unit(v):
 b=int(np.asarray(v,dtype=np.float32).view(np.uint32));sign=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255:raise ValueError("nonfinite F32")
 return sign*f if e==0 else sign*((1<<23)|f)<<(e-1)
def det6(t,u):
 p0,p1,p2,p3=(u[int(i)] for i in t);a=[p1[j]-p0[j] for j in range(3)];b=[p2[j]-p0[j] for j in range(3)];c=[p3[j]-p0[j] for j in range(3)]
 return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def quality(p,t):
 bad=[];qmin=float("inf");vmin=float("inf")
 for lo in range(0,len(t),50000):
  x=np.asarray(p[np.asarray(t[lo:lo+50000],np.int64)],np.float64);a=x[:,1]-x[:,0];b=x[:,2]-x[:,0];c=x[:,3]-x[:,0];d=np.einsum("ij,ij->i",a,np.cross(b,c));es=[]
  for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
   e=x[:,i]-x[:,j];es.append(np.einsum("ij,ij->i",e,e))
  mx=np.maximum.reduce(es);q=np.abs(d)/np.power(mx,1.5);v=np.abs(d)/6;bad.append((v<=1e-18)|(q<1e-5));qmin=min(qmin,float(q.min()));vmin=min(vmin,float(v.min()))
 return np.concatenate(bad),qmin,vmin
def main():
 start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 paths={"input":I,"output":O,"transfer":TR,"opmap":OM,"surface_input":SI,"surface_output":SO,"surface_report":SR,"surface_ledger":SL,"protection":P}
 actual={k:sha(v) for k,v in paths.items()}
 for k,v in PIN.items():need(actual[k]==v,k+" SHA mismatch")
 x=load(I);y=load(O);si=load(SI);so=load(SO);prot=load(P);tr=json.loads(TR.read_text());sr=json.loads(SR.read_text())
 need(tr["status"]=="PASS_D500_INDEXED379_VOLUME_QUOTIENT_PROPOSAL","transfer not PASS")
 need(sr["status"]=="PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL","surface not PASS")
 p=np.asarray(x["points_world_m_f32"],np.float32);t=np.asarray(x["tetrahedra"],np.int64);m=np.asarray(x["material_indices"],np.int32);op=np.asarray(y["points_world_m_f32"],np.float32);ot=np.asarray(y["tetrahedra"],np.int64);om=np.asarray(y["material_indices"],np.int32)
 need(np.array_equal(p,op) and np.array_equal(x["points_world_m"],p) and np.array_equal(y["points_world_m"],op),"point bytes/views changed")
 need(np.all(m==16) and np.all(om==16),"material changed")
 node=np.asarray(y["input_node_to_output_node"],np.int64);need(node.shape==(len(p),),"node-map shape")
 need(np.array_equal(y["output_node_to_input_node"],np.arange(len(p))),"point domain compacted")
 raw=[json.loads(z) for z in SL.read_text().splitlines() if z.strip()];vops=[json.loads(z) for z in OM.read_text().splitlines() if z.strip()]
 need(len(raw)==len(vops)==1277,"operation count mismatch")
 dmap=np.asarray(x["fat_current_target_rebind_D_point_to_volume_rows"],np.int64);need(len(dmap)==len(si["points_world_m_f32"]),"D map length")
 need(np.array_equal(p[dmap],si["points_world_m_f32"]),"D point map not bitwise")
 drops=[];keeps=[];allrows=[]
 for i,(a,b) in enumerate(zip(raw,vops)):
  need(int(b["surface_operation_index"])==i,"operation order mismatch")
  dr=int(a["collapsed_input_point_row"]);kr=int(a["retained_input_point_row"]);need((dr,kr)==(int(b["drop_surface_row"]),int(b["keep_surface_row"])),"surface op mismatch")
  dv=int(b["drop_volume_row"]);kv=int(b["keep_volume_row"]);need((dv,kv)==(int(dmap[dr]),int(dmap[kr])),"volume endpoint mismatch")
  need(int(node[dv])==kv,"node quotient mismatch")
  need(dv not in drops and dv not in keeps and kv not in drops and kv not in keeps,"operation endpoints reused")
  drops.append(dv);keeps.append(kv)
  rows=np.asarray(b["affected_input_cell_rows"],np.int64);deleted=np.asarray(b["deleted_input_cell_rows"],np.int64)
  need(len(rows)>0 and len(np.unique(rows))==len(rows),"bad affected rows")
  need(set(map(int,deleted)).issubset(set(map(int,rows))),"deleted rows outside star")
  need(int(b["removed_cell_count"])==len(deleted),"deleted count mismatch")
  allrows.extend(map(int,rows))
 need(np.array_equal(np.sort(np.flatnonzero(node!=np.arange(len(node)))),np.sort(drops)),"unexpected node remap")
 need(np.array_equal(node[np.asarray(drops)],np.asarray(keeps)),"node remap endpoints mismatch")
 flat=t.reshape(-1);cell=np.repeat(np.arange(len(t),dtype=np.int64),4);order=np.argsort(flat);counts=np.bincount(flat,minlength=len(p));off=np.r_[0,np.cumsum(counts)]
 for i,b in enumerate(vops):
  dv=int(b["drop_volume_row"]);star=np.sort(cell[order[off[dv]:off[dv+1]]]);given=np.sort(np.asarray(b["affected_input_cell_rows"],np.int64))
  need(np.array_equal(star,given),f"operation {i} not full old drop-node star")
  kv=int(b["keep_volume_row"]);d=(np.any(t[star]==dv,axis=1)&np.any(t[star]==kv,axis=1));need(np.array_equal(np.sort(star[d]),np.sort(np.asarray(b["deleted_input_cell_rows"],np.int64))),f"operation {i} deleted rows mismatch")
 affected=np.unique(np.asarray(allrows,np.int64));protrows=np.asarray(prot["protected_current_cell_rows"],np.int64)
 need(len(np.intersect1d(affected,protrows))==0,"operation intersects protected comp0/comp1 stars")
 mt=node[t];smt=np.sort(mt,axis=1);deg=np.any(smt[:,1:]==smt[:,:-1],axis=1);ret=np.flatnonzero(~deg)
 cm=np.asarray(y["input_cell_to_output_cell"],np.int64);out2in=np.asarray(y["output_cell_to_input_cell"],np.int64)
 need(np.array_equal(cm,np.where(deg,-1,np.cumsum(~deg)-1)),"input/output cell map mismatch")
 need(np.array_equal(np.asarray(y["retained_input_cell_rows"]),ret) and np.array_equal(out2in,ret),"retained cell map mismatch")
 need(np.array_equal(ot,mt[ret]) and np.array_equal(om,m[ret]),"candidate does not reconstruct from point quotient")
 need(np.all(cm[protrows]>=0) and np.array_equal(ot[cm[protrows]],t[protrows]) and np.array_equal(om[cm[protrows]],m[protrows]),"protected comp0/1 rows changed")
 units=[tuple(unit(c) for c in row) for row in p]
 exact_input_min=None
 for lo in range(0,len(t),10000):
  for j,tet in enumerate(t[lo:lo+10000]):
   d=det6(tet,units)
   if d<=0:raise ValueError(f"input exact sign fail row {lo+j}")
   exact_input_min=d if exact_input_min is None or d<exact_input_min else exact_input_min
 oldsum=0;newsum=0;deleted_union=0;newpos=0
 for row in affected:
  old=t[row];new=node[old];dold=det6(old,units);need(dold>0,"affected input cell not exact-positive");oldsum+=dold
  if len(set(map(int,new)))<4:deleted_union+=1;continue
  dnew=det6(new,units);need(dnew>0,"replacement output cell not exact-positive");newsum+=dnew;newpos+=1
 delta=newsum-oldsum
 need(str(delta)==tr["sequential_exact_det6_delta_sum_integer"],"exact affected-union delta mismatch")
 need(sum(int(b["exact_det6_delta_integer"]) for b in vops)==delta,"operation det deltas do not sum")
 need(deleted_union==int(tr["removed_cell_count"]),"removed cell count mismatch")
 oldbad,q0,v0=quality(p,t);newbad,q1,v1=quality(op,ot)
 need(int(oldbad.sum())==int(tr["native_bad_cells_before"]) and int(newbad.sum())==int(tr["native_bad_cells_after"]),"native quality counts mismatch")
 regress=int(np.count_nonzero((~oldbad[ret])&newbad));need(regress==0,"old-good cell regression")
 b0=boundary(t);b1=boundary(ot)
 for got,stored,label in ((chain(b0),chain(x["boundary_faces"]),"input"),(chain(b1),chain(y["boundary_faces"]),"output")):
  need(np.array_equal(got[0],stored[0]) and np.array_equal(got[1],stored[1]),label+" stored boundary mismatch")
 mapped_boundary=node[np.asarray(x["boundary_faces"],np.int64)]
 mapped_target=node[np.asarray(x["desired_boundary_faces"],np.int64)]
 smtgt=np.sort(mapped_target,axis=1);targetdeg=np.any(smtgt[:,1:]==smtgt[:,:-1],axis=1);mapped_target=mapped_target[~targetdeg]
 proposed=np.asarray(y["proposed_current_desired_boundary_faces"],np.int64)
 need(np.array_equal(chain(mapped_boundary)[0],chain(b1)[0]) and np.array_equal(chain(mapped_boundary)[1],chain(b1)[1]),"mapped source boundary mismatch")
 need(np.array_equal(chain(mapped_target)[0],chain(proposed)[0]) and np.array_equal(chain(mapped_target)[1],chain(proposed)[1]),"mapped target construction mismatch")
 need(np.array_equal(chain(b1)[0],chain(proposed)[0]) and np.array_equal(chain(b1)[1],chain(proposed)[1]),"output boundary differs from target")
 closed(b1,"output boundary");closed(proposed,"target")
 report={"schema":"h706.independent-d500-indexed379-replay.v1","status":"PASS_INDEPENDENT_D500_VOLUME_REPLAY","pins":{k:{"path":str(v),"sha256":actual[k]} for k,v in paths.items()},"input_cells":len(t),"output_cells":len(ot),"removed_cells":len(t)-len(ot),"input_points":len(p),"output_points":len(op),"all_point_bytes_preserved":True,"all_input_exact_positive_cells_recomputed":len(t),"all_input_exact_positive":True,"minimum_input_det6_integer":str(exact_input_min),"operation_count":len(vops),"affected_full_star_cell_rows":len(affected),"protected_comp0_comp1_cells":len(protrows),"protected_cells_unchanged":True,"all_operations_disjoint_from_protected_stars":True,"all_output_tets_reconstructed_from_input_quotient":True,"all_material_indices_16":True,"affected_union_exact_positive_replacements":newpos,"affected_union_deleted_cells":deleted_union,"exact_det6_delta_integer":str(delta),"exact_volume_delta_m3":str(delta/(6*(1<<447))),"native_bad_cells_recomputed_before":int(oldbad.sum()),"native_bad_cells_recomputed_after":int(newbad.sum()),"native_bad_cell_gain":int(oldbad.sum()-newbad.sum()),"old_good_to_bad_regressions":regress,"minimum_normalized_det_before":q0,"minimum_normalized_det_after":q1,"minimum_cell_volume_before_m3":v0,"minimum_cell_volume_after_m3":v1,"input_and_output_boundaries_reconstructed":True,"output_boundary_equals_mapped_input_and_target":True,"target_missing_oriented_records":0,"target_extra_oriented_records":0,"mass_mapping":"NOT_RUN","native_steps":0,"physical_admission":False,"elapsed_seconds":time.monotonic()-start,"limitations":["D-only geometric volume quotient; paired I and fan-composed branch merges require exact-key replay.","Mass mapping, final combined source context and native installation remain unqualified."]}
 q=OUT/"independent-verification.json";q.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"report":str(q),"sha256":sha(q),"input_output_cells":[len(t),len(ot)],"operations":len(vops),"native_bad":[int(oldbad.sum()),int(newbad.sum())],"exact_delta":str(delta),"elapsed_seconds":round(report["elapsed_seconds"],3)},sort_keys=True))
def fail(exc):
 OUT.mkdir(parents=True,exist_ok=True);q=OUT/"failure.json";r={"schema":"h706.independent-d500-indexed379-replay.v1","status":"FAIL_INDEPENDENT_D500_VOLUME_REPLAY","error":repr(exc),"traceback":traceback.format_exc()};q.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n");print(json.dumps({"status":r["status"],"report":str(q),"sha256":sha(q),"error":r["error"]},sort_keys=True))
try:main()
except Exception as e:fail(e);raise
