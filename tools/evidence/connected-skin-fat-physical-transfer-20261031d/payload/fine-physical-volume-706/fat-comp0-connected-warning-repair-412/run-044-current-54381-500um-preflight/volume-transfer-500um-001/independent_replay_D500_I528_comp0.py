#!/usr/bin/env python3
"""Independent replay of the exact-key D500+I528 composition on comp0."""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/volume-transfer-500um-001"
BASE=H/"fat-comp0-connected-warning-repair-412/run-042-current-54381-normalized-rebind/fat-current-typed-target-rebound-54381-lineage-corrected.npz"
DOUT=R/"run001/fat-D500-short-edge-quotient-volume-proposal.npz"; DREPORT=R/"run001/report.json"; DOPMAP=R/"run001/operation-volume-star-map.jsonl"
COMP=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-007-kernel-fan-composed/fat-comp0-kernel-fan-composed-volume-proposal.npz"
PROT=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/run-038-kernel-fans-297/protection-007/comp0-comp1-protected-stars.npz"
IINPUT=H/"paired-I-current-protected-528-root/exchange520/fat-coupled-I-quotient-volume-proposal.npz"; IREPORT=H/"paired-I-current-protected-528-root/exchange520/report.json"
ITARGET=H/"paired-I-current-protected-528-root/fat-target397/fat-current-typed-target-rebound.npz"; ITREPORT=ITARGET.parent/"report.json"
CAND=R/"compose-D500-I528-comp0-54138-005/fat-D500-I528-comp0-kernel-fan-composed-volume-proposal.npz"; CREPORT=CAND.parent/"report.json"
OUT=R/"compose-D500-I528-comp0-54138-005/independent-replay-001"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def read(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def need(x,m):
 if not x:raise ValueError(m)
def chain(f):
 a=np.asarray(f,np.int64).reshape((-1,3)); good=np.all(np.diff(np.sort(a,axis=1),axis=1)!=0,axis=1);a=a[good]
 s=np.sort(a,axis=1);inv=(a[:,0]>a[:,1]).astype(np.int8)+(a[:,0]>a[:,2]).astype(np.int8)+(a[:,1]>a[:,2]).astype(np.int8);sg=(1-2*(inv&1)).astype(np.int64);k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];return k[st],np.add.reduceat(sg,st)
def boundary(t):
 F=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64);f=np.asarray(t,np.int64)[:,F].reshape((-1,3));s=np.sort(f,axis=1);inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8);sg=(1-2*(inv&1)).astype(np.int8);k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];ct=np.diff(np.r_[st,len(k)]);su=np.add.reduceat(sg,st)
 if np.any(ct>2) or np.any((ct==2)&(su!=0)):raise ValueError("nonmanifold output face table")
 return f[o[st[ct==1]]]
def qstats(p,t):
 bad=[];qmins=[];vmins=[]
 for lo in range(0,len(t),50000):
  x=np.asarray(p[t[lo:lo+50000]],np.float64);a=x[:,1]-x[:,0];b=x[:,2]-x[:,0];c=x[:,3]-x[:,0];det=np.einsum("ij,ij->i",a,np.cross(b,c));es=[]
  for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
   e=x[:,i]-x[:,j];es.append(np.einsum("ij,ij->i",e,e))
  mx=np.maximum.reduce(es);q=np.abs(det)/np.power(mx,1.5);v=np.abs(det)/6.;bad.append((v<=1e-18)|(q<1e-5));qmins.append(float(q.min()));vmins.append(float(v.min()))
 return np.concatenate(bad),min(qmins),min(vmins)
def keys(t,m):
 s=np.sort(np.asarray(t,np.int64),axis=1);return np.rec.fromarrays([np.asarray(m,np.int32),s[:,0],s[:,1],s[:,2],s[:,3]],names="mat,a,b,c,d")
def unit(v):
 b=int(np.asarray(v,dtype=np.float32).view(np.uint32));sgn=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255:raise ValueError("nonfinite point")
 return sgn*f if e==0 else sgn*((1<<23)|f)<<(e-1)
def det6(t,u):
 p0,p1,p2,p3=(u[int(i)] for i in t);a=[p1[j]-p0[j] for j in range(3)];b=[p2[j]-p0[j] for j in range(3)];c=[p3[j]-p0[j] for j in range(3)]
 return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def main():
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 base=read(BASE);d=read(DOUT);comp=read(COMP);pr=read(PROT);i=read(IINPUT);target0=read(ITARGET);out=read(CAND)
 dr=json.loads(DREPORT.read_text());ir=json.loads(IREPORT.read_text());tr=json.loads(ITREPORT.read_text());cr=json.loads(CREPORT.read_text())
 iops=[x for x in ir["roles"] if x["role"]=="Fat"][0]["operations"]
 pins={str(p):sha(p) for p in (BASE,DOUT,DREPORT,DOPMAP,COMP,PROT,IINPUT,IREPORT,ITARGET,ITREPORT,CAND,CREPORT)}
 expected={str(BASE):"5aa02086aac4b20e879332cd9d5ba64da636dd8a520de54e7b96b00225c7fe42",str(DOUT):"35a5545ed4d3bbe40499df0b143456683fb8f0739372e2a53b8a3227e4474b24",str(COMP):"54138c6a3cbc62ff37a0bf9d86257896587304a493f9e05f9c9ec02fc3dc80fb",str(IINPUT):"08d1a9b27ab9407e66d7a3912ebdadf46787fd5ad9f519378c20f1d8503f98d6",str(IREPORT):"339c2045cfef9a76005fb2bc0eb4bb973ec73015467ab74dd039a4c4d5b34773",str(ITARGET):"d36c7ab4987e55a160c4a9f9f371f44e67b1396239811e2e9b6f30b349dc2085",str(CAND):"d899b2789c0d4fc91553a5d46b1bb8d7b40e9c818d6301f6b32a3265976e3703"}
 for p,h in expected.items():need(pins[p]==h,"pinned input/output hash mismatch "+p)
 bp=base["points_world_m_f32"];bt=base["tetrahedra"];bm=base["material_indices"];cp=comp["points_world_m_f32"];ct=comp["tetrahedra"];cm=comp["material_indices"]
 need(np.array_equal(out["points_world_m_f32"],cp) and np.array_equal(out["points_world_m"],cp),"output point arrays differ from comp0 parent")
 need(np.array_equal(bp,cp[:len(bp)]) and np.array_equal(bp,d["points_world_m_f32"]) and np.array_equal(bp,i["points_world_m_f32"]),"source point prefix changed")
 dmap=np.asarray(d["input_node_to_output_node"],np.int64);nd=len(bp);need(len(dmap)==nd,"D map source row domain mismatch")
 idrop=np.asarray([o["retired_volume_node"] for o in iops],np.int64);ikeep=np.asarray([o["kept_volume_node"] for o in iops],np.int64)
 inode=np.arange(nd,dtype=np.int64);inode[idrop]=ikeep
 combined=np.arange(len(cp),dtype=np.int64);combined[:nd]=dmap
 need(np.array_equal(combined[idrop],idrop),"D and I map points overlap")
 combined[idrop]=ikeep
 need(np.array_equal(out["input_node_to_output_node"],combined),"composed input node map differs from independently replayed node maps")
 mapped=combined[ct];s=np.sort(mapped,axis=1);deg=np.any(s[:,1:]==s[:,:-1],axis=1);keep=np.flatnonzero(~deg);reconstructed=mapped[keep]
 need(np.array_equal(out["tetrahedra"],reconstructed),"output tet rows differ from exact D500+I528 comp0 replay")
 need(np.array_equal(out["material_indices"],cm[keep]),"output material rows differ")
 need(len(ct)-len(keep)==int(dr["removed_cell_count"])+sum(int(o["removed_cells"]) for o in iops),"removed cell total differs from operation ledgers")
 # Independently reconstruct exact source-cell-key joins and prove both operation star sets are disjoint.
 def mapkeys(rows):
  bk=keys(bt,bm);ck=keys(ct,cm);order=np.argsort(ck,kind="stable");sk=ck[order];q=bk[np.asarray(rows,np.int64)];pos=np.searchsorted(sk,q);hit=pos<len(sk);hit[hit]&=(sk[pos[hit]]==q[hit]);need(np.all(hit),"operation source key missing on comp0 branch");return order[pos],q
 dop=[json.loads(x) for x in DOPMAP.read_text().splitlines() if x.strip()];drows=np.unique(np.concatenate([np.asarray(o["affected_input_cell_rows"],np.int64) for o in dop]));irows=np.unique(np.concatenate([np.asarray(o["source_cell_rows"],np.int64) for o in iops]))
 dcomp,dk=mapkeys(drows);icomp,ik=mapkeys(irows)
 need(not np.intersect1d(drows,irows).size and not np.intersect1d(dcomp,icomp).size,"D/I source stars overlap")
 with np.load(PROT,allow_pickle=False) as p:
  pkeys=p["protected_tet_material_corner_keys"];pp=p["protected_volume_point_rows"]
 need(not np.intersect1d(dk,pkeys).size and not np.intersect1d(ik,pkeys).size,"D/I operation keys touch comp0/1 protection")
 ddrop=np.asarray([o["drop_volume_row"] for o in dop],np.int64);dkeep=np.asarray([o["keep_volume_row"] for o in dop],np.int64)
 need(not np.intersect1d(np.r_[ddrop,idrop],np.r_[dkeep,ikeep]).size,"D/I drop/keep endpoints conflict")
 need(np.array_equal(combined[pp],pp),"protected point map changed")
 incident=np.flatnonzero(np.isin(ct,np.r_[ddrop,idrop]).any(axis=1));need(np.array_equal(np.sort(incident),np.sort(np.r_[dcomp,icomp])),"full drop-point star does not equal operation source ledger")
 # Independently derive the combined typed target from the current I528 target, then D500 node quotient.
 tfaces=np.asarray(target0["desired_boundary_faces"],np.int64);mapped_target=combined[tfaces];ss=np.sort(mapped_target,axis=1);td=np.any(ss[:,1:]==ss[:,:-1],axis=1);tf=mapped_target[~td]
 for k in ("desired_boundary_faces","proposed_current_desired_boundary_faces","mapped_current_boundary_faces"):
  a=np.asarray(out[k],np.int64)
  need(np.array_equal(chain(a)[0],chain(tf)[0]) and np.array_equal(chain(a)[1],chain(tf)[1]),"output target/metadata differs from independent chain")
 bo=boundary(reconstructed);need(np.array_equal(chain(bo)[0],chain(tf)[0]) and np.array_equal(chain(bo)[1],chain(tf)[1]),"independent actual boundary differs from target")
 need(len(bo)==len(tf)==459940,"unexpected target face count")
 # Exact integer lattice determinant replay for every output cell.
 units=[tuple(unit(x) for x in row) for row in cp];min_det=None
 for row,tet in enumerate(reconstructed):
  z=det6(tet,units)
  if z<=0:raise ValueError(f"nonpositive exact output tet {row}")
  if min_det is None or z<min_det:min_det=z
 # Recompute native quality counts and the explicitly authorized I528 exchange witnesses.
 oldbad,oldq,oldv=qstats(cp,ct);newbad,newq,newv=qstats(cp,reconstructed)
 dmapext=np.arange(len(cp),dtype=np.int64);dmapext[:nd]=dmap
 dt=dmapext[ct];ds=np.sort(dt,axis=1);dd=np.any(ds[:,1:]==ds[:,:-1],axis=1);dkeep_cells=np.flatnonzero(~dd);dbad,dq,dv=qstats(cp,dt[dkeep_cells])
 dreg=int(np.count_nonzero((~oldbad[dkeep_cells])&dbad));need(dreg==0,"D500 replays an old-good regression")
 outbad_map=np.zeros(len(ct),bool);outbad_map[keep]=newbad
 regs=keep[(~outbad_map[keep] if False else np.zeros(len(keep),bool))] # overwritten below from D-only row mask
 dflags=np.zeros(len(ct),bool);dflags[dkeep_cells]=dbad
 regs=keep[(~dflags[keep])&newbad]
 actual_base=np.asarray(out["D500_I528_quality_exchange_regression_base54381_rows"],np.int64)
 actual_comp=np.asarray(out["D500_I528_quality_exchange_regression_comp0_input_rows"],np.int64)
 actual_ops=np.asarray(out["D500_I528_quality_exchange_regression_I528_operation_indices"],np.int64)
 # Reconstruct the affected-row map by exact tet keys, then require exact one-to-one witness correspondence.
 ckeys=keys(ct,cm);ordc=np.argsort(ckeys,kind="stable");sck=ckeys[ordc];bkeys=keys(bt,bm)
 brpos=np.searchsorted(sck,bkeys);bhit=brpos<len(sck);bhit[bhit]&=(sck[brpos[bhit]]==bkeys[bhit]);base_to_comp=np.full(len(bt),-1,np.int64);base_to_comp[bhit]=ordc[brpos[bhit]]
 comp_to_base={int(v):int(k) for k,v in enumerate(base_to_comp) if v>=0}
 reg_base=np.asarray([comp_to_base[int(x)] for x in regs],np.int64)
 op_for={}
 authorized={}
 for oi,op in enumerate(iops):
  for r in op["source_cell_rows"]:op_for[int(r)]=oi
  for r in op.get("new_retained_quality_failure_rows",[]):authorized[int(r)]=oi
 need(np.array_equal(np.sort(regs),np.sort(actual_comp)),"reported I quality debt rows differ from independently recomputed rows")
 need(np.array_equal(np.sort(reg_base),np.sort(actual_base)),"reported I quality debt source rows differ from independent exact-key map")
 need(len(authorized)==8 and set(map(int,reg_base))==set(authorized),"I528 old-good-to-bad rows are not exactly authorized source witnesses")
 derived_ops=np.asarray([authorized[int(r)] for r in reg_base],np.int64)
 # Match each witness operation by source ledger and explicitly recorded new-retained row.
 actual_op_by_row={int(r):int(o) for r,o in zip(actual_base,actual_ops)}
 need(all(actual_op_by_row[int(r)]==authorized[int(r)] for r in reg_base),"I528 quality exchange operation IDs mismatch")
 need(int(dbad.sum())==14677 and int(newbad.sum())==14366 and int(dbad.sum()-newbad.sum())==311,"quality counts differ from source reports")
 need(int(oldbad.sum())==16146 and dreg==0,"comp0 or D500 source quality baseline mismatch")
 # Exact det6 volume delta over all changed source stars.
 stars=np.unique(np.r_[drows,irows]);r_to_c={int(r):int(crow) for r,crow in zip(drows,dcomp)};r_to_c.update({int(r):int(crow) for r,crow in zip(irows,icomp)})
 touched=np.asarray([r_to_c[int(r)] for r in stars],np.int64);delta=0
 for row in touched:
  old=det6(ct[int(row)],units);newtet=combined[ct[int(row)]];new=0 if len(set(map(int,newtet)))<4 else det6(newtet,units);delta+=new-old
 need(str(delta)==str(out["operation_exact_det6_delta_integer"].astype("U256").astype(object).astype(str).tolist() and sum(int(x) for x in out["operation_exact_det6_delta_integer"])),"exact volume delta ledger mismatch")
 need(delta==int(json.loads(CREPORT.read_text()).get("exact_volume_delta",{}).get("det6_integer",delta)) if "exact_volume_delta" in json.loads(CREPORT.read_text()) else True,"composition report exact delta mismatch")
 receipt={
  "schema":"h706.d500-i528-comp0-independent-replay.v1","status":"PASS_INDEPENDENT_D500_I528_COMPOSITION_REPLAY",
  "candidate":{"path":str(CAND),"sha256":pins[str(CAND)]},"candidate_report":{"path":str(CREPORT),"sha256":pins[str(CREPORT)]},
  "inputs":{str(p):{"sha256":pins[str(p)]} for p in (BASE,DOUT,COMP,PROT,IINPUT,IREPORT,ITARGET,ITREPORT)},
  "replay":{"input_comp0_cells":len(ct),"output_cells":len(reconstructed),"removed_cells":int(deg.sum()),"input_points":len(cp),"point_arrays_bit_identical":True,"exact_key_join_D_cells":len(drows),"exact_key_join_I_cells":len(irows),"D_I_star_overlap":0,"protected_key_overlap":0,"all_output_exact_positive":True,"minimum_exact_det6_integer":str(min_det),"target_faces":len(tf),"oriented_boundary_mismatch":0,"volume_delta_det6_integer":str(delta)},
  "quality":{"input_bad":int(oldbad.sum()),"D500_only_bad":int(dbad.sum()),"D500_old_good_regressions":dreg,"combined_bad":int(newbad.sum()),"I528_authorized_old_good_to_bad":len(regs),"I528_witness_rows":sorted(map(int,reg_base)),"I528_witness_ops":[int(authorized[int(r)]) for r in sorted(map(int,reg_base))],"I528_net_descent":int(dbad.sum()-newbad.sum())},
  "physical_admission":False,"mass_mapping":"NOT_RUN","native_steps":0,"elapsed_seconds":time.monotonic()-start
 }
 (OUT/"independent-verification.json").write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":receipt["status"],"candidate_sha256":pins[str(CAND)],"receipt":str(OUT/"independent-verification.json"),"receipt_sha256":sha(OUT/"independent-verification.json"),"quality":[int(oldbad.sum()),int(dbad.sum()),int(newbad.sum())],"authorized_regressions":len(regs),"delta":str(delta),"elapsed_s":round(receipt["elapsed_seconds"],3)},sort_keys=True))
if __name__=="__main__":main()

