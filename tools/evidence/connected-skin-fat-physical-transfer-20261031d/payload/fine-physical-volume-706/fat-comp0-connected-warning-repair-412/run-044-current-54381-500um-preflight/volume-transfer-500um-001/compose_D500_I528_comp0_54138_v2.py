#!/usr/bin/env python3
"""Exact-key composition of D500 and I528 endpoint quotients on the comp0 fan branch."""
from pathlib import Path
import hashlib, json, time, traceback
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
ROOT=H/"fat-comp0-connected-warning-repair-412"
BASE=ROOT/"run-042-current-54381-normalized-rebind/fat-current-typed-target-rebound-54381-lineage-corrected.npz"
DOUT=ROOT/"run-044-current-54381-500um-preflight/volume-transfer-500um-001/run001/fat-D500-short-edge-quotient-volume-proposal.npz"
DREPORT=DOUT.parent/"report.json"
DOPMAP=DOUT.parent/"operation-volume-star-map.jsonl"
DSURF=ROOT/"run-044-current-54381-500um-preflight/surface-proposal-500um/candidate-D-short-edge-collapse-raw.npz"
DSURFREPORT=DSURF.parent/"report.json"
COMP=ROOT/"run-037-local-conditioning-492/result-007-kernel-fan-composed/fat-comp0-kernel-fan-composed-volume-proposal.npz"
COMPREPORT=COMP.parent/"report.json"
PROT=ROOT/"run-037-local-conditioning-492/run-038-kernel-fans-297/protection-007/comp0-comp1-protected-stars.npz"
IPROOT=H/"paired-I-current-protected-528-root"
IINPUT=IPROOT/"exchange520/fat-coupled-I-quotient-volume-proposal.npz"
IREPORT=IPROOT/"exchange520/report.json"
ITARGET=IPROOT/"fat-target397/fat-current-typed-target-rebound.npz"
ITARGETREPORT=IPROOT/"fat-target397/report.json"
ITYPED=IPROOT/"typed463001/candidate-I-typed-current.npz"
ITYPEDRECEIPT=IPROOT/"typed463001/receipt.json"
OUTDIR=ROOT/"run-044-current-54381-500um-preflight/volume-transfer-500um-001/compose-D500-I528-comp0-54138-002"
CANDIDATE=OUTDIR/"fat-D500-I528-comp0-kernel-fan-composed-volume-proposal.npz"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z: return {k:np.array(z[k],copy=True) for k in z.files}
def need(x,msg):
 if not x: raise ValueError(msg)
def chain(f):
 a=np.asarray(f,np.int64).reshape((-1,3))
 keep=np.all(np.diff(np.sort(a,axis=1),axis=1)!=0,axis=1);a=a[keep]
 if not len(a): return np.empty(0,np.int64),np.empty(0,np.int64)
 s=np.sort(a,axis=1); inv=(a[:,0]>a[:,1]).astype(np.int8)+(a[:,0]>a[:,2]).astype(np.int8)+(a[:,1]>a[:,2]).astype(np.int8)
 sg=(1-2*(inv&1)).astype(np.int64);k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o]
 st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];su=np.add.reduceat(sg,st);kk=k[st];ok=su!=0
 return kk[ok],su[ok]
def boundary(t):
 F=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
 f=np.asarray(t,np.int64)[:,F].reshape((-1,3));s=np.sort(f,axis=1)
 inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8)
 sg=(1-2*(inv&1)).astype(np.int8);k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o]
 st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];ct=np.diff(np.r_[st,len(k)]);su=np.add.reduceat(sg,st)
 if np.any(ct>2) or np.any((ct==2)&(su!=0)): raise ValueError("nonmanifold or inconsistent tet face")
 return f[o[st[ct==1]]]
def closed(f,label):
 a=np.asarray(f,np.int64);e=np.concatenate((a[:,[0,1]],a[:,[1,2]],a[:,[2,0]]));sg=np.where(e[:,0]<e[:,1],1,-1).astype(np.int8);s=np.sort(e,axis=1);k=s[:,0]*(1<<20)+s[:,1];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1]
 if np.any(np.add.reduceat(sg,st)): raise ValueError(label+" not closed")
def keys(t,m):
 s=np.sort(np.asarray(t,np.int64),axis=1)
 return np.rec.fromarrays([np.asarray(m,np.int32),s[:,0],s[:,1],s[:,2],s[:,3]],names="mat,a,b,c,d")
def quality(p,t):
 bad=[];qmin=float("inf");vmin=float("inf")
 for lo in range(0,len(t),50000):
  x=np.asarray(p[np.asarray(t[lo:lo+50000],np.int64)],np.float64);a=x[:,1]-x[:,0];b=x[:,2]-x[:,0];c=x[:,3]-x[:,0];det=np.einsum("ij,ij->i",a,np.cross(b,c));es=[]
  for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
   e=x[:,i]-x[:,j];es.append(np.einsum("ij,ij->i",e,e))
  mx=np.maximum.reduce(es);q=np.abs(det)/np.power(mx,1.5);v=np.abs(det)/6.
  bad.append((v<=1e-18)|(q<1e-5));qmin=min(qmin,float(q.min()));vmin=min(vmin,float(v.min()))
 return np.concatenate(bad),qmin,vmin
def unit(v):
 b=int(np.asarray(v,dtype=np.float32).view(np.uint32));sign=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255: raise ValueError("nonfinite F32 point")
 return sign*f if e==0 else sign*((1<<23)|f)<<(e-1)
def det6(t,u):
 p0,p1,p2,p3=(u[int(i)] for i in t);a=[p1[j]-p0[j] for j in range(3)];b=[p2[j]-p0[j] for j in range(3)];c=[p3[j]-p0[j] for j in range(3)]
 return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])

def main():
 OUTDIR.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 b=load(BASE);d=load(DOUT);c=load(COMP);pr=load(PROT);i=load(IINPUT);t=load(ITARGET)
 dr=json.loads(DREPORT.read_text());ir=json.loads(IREPORT.read_text());cr=json.loads(COMPREPORT.read_text());tr=json.loads(ITARGETREPORT.read_text())
 ro=[x for x in ir["roles"] if x["role"]=="Fat"][0];iops=ro["operations"]
 pins={str(p):sha(p) for p in (BASE,DOUT,DREPORT,DOPMAP,DSURF,DSURFREPORT,COMP,COMPREPORT,PROT,IINPUT,IREPORT,ITARGET,ITARGETREPORT,ITYPED,ITYPEDRECEIPT)}
 expected={
  str(BASE):"5aa02086aac4b20e879332cd9d5ba64da636dd8a520de54e7b96b00225c7fe42",
  str(DOUT):"35a5545ed4d3bbe40499df0b143456683fb8f0739372e2a53b8a3227e4474b24",
  str(COMP):"54138c6a3cbc62ff37a0bf9d86257896587304a493f9e05f9c9ec02fc3dc80fb",
  str(IINPUT):"08d1a9b27ab9407e66d7a3912ebdadf46787fd5ad9f519378c20f1d8503f98d6",
  str(IREPORT):"339c2045cfef9a76005fb2bc0eb4bb973ec73015467ab74dd039a4c4d5b34773",
  str(ITARGET):"d36c7ab4987e55a160c4a9f9f371f44e67b1396239811e2e9b6f30b349dc2085",
  str(ITYPED):"6f59262c2ae70065576166dd84f767afb64352241e76009707f78447afe477d1",
 }
 for p,h in expected.items(): need(pins[p]==h,"input pin mismatch: "+p)
 bp=np.asarray(b["points_world_m_f32"],np.float32);bt=np.asarray(b["tetrahedra"],np.int64);bm=np.asarray(b["material_indices"],np.int32)
 cp=np.asarray(c["points_world_m_f32"],np.float32);ct=np.asarray(c["tetrahedra"],np.int64);cm=np.asarray(c["material_indices"],np.int32)
 dp=np.asarray(d["points_world_m_f32"],np.float32);dt=np.asarray(d["tetrahedra"],np.int64);dm=np.asarray(d["material_indices"],np.int32);dnode=np.asarray(d["input_node_to_output_node"],np.int64)
 ip=np.asarray(i["points_world_m_f32"],np.float32);it=np.asarray(i["tetrahedra"],np.int64);im=np.asarray(i["material_indices"],np.int32)
 need(np.array_equal(bp,dp) and np.array_equal(bp,ip),"D/I quotient candidate point coordinates differ from 54381 base")
 need(len(cp)>=len(bp) and np.array_equal(bp,cp[:len(bp)]) and np.array_equal(c["points_world_m"],cp),"comp0 point prefix mismatch")
 nbase=len(bp);need(len(dnode)==nbase,"D point map domain mismatch")
 need(len(iops)==267 and int(ir["passing_operations"])==267,"I528 operation count mismatch")
 idrop=np.asarray([o["retired_volume_node"] for o in iops],np.int64); ikeep=np.asarray([o["kept_volume_node"] for o in iops],np.int64)
 need(len(np.unique(idrop))==len(idrop) and len(np.unique(ikeep))==len(ikeep),"I528 endpoint ids repeat")
 need(not np.intersect1d(idrop,ikeep).size,"I528 drop/keep sets overlap")
 need(np.all((idrop>=0)&(idrop<nbase)&(ikeep>=0)&(ikeep<nbase)),"I528 node index out of range")
 # Independently reconstruct I528 from its frozen operation ledger and exact source row domain.
 imap=np.arange(nbase,dtype=np.int64);imap[idrop]=ikeep
 itmap=imap[bt];its=np.sort(itmap,axis=1);itdeg=np.any(its[:,1:]==its[:,:-1],axis=1)
 need(int(itdeg.sum())==sum(int(o["removed_cells"]) for o in iops),"I528 operation removed-cell tally mismatch")
 irecon=itmap[~itdeg];need(np.array_equal(irecon,it),"I528 operations do not exactly reproduce candidate tet table")
 need(np.array_equal(im,np.asarray(b["material_indices"],np.int32)[~itdeg]),"I528 material row replay mismatch")
 need(np.array_equal(ip,bp),"I528 candidate changed point coordinates")
 dop=[json.loads(x) for x in DOPMAP.read_text().splitlines() if x.strip()]
 need(len(dop)==1277 and int(dr["transferred_operation_count"])==1277,"D500 operation count mismatch")
 ddrop=np.asarray([o["drop_volume_row"] for o in dop],np.int64);dkeep=np.asarray([o["keep_volume_row"] for o in dop],np.int64)
 drows=np.unique(np.concatenate([np.asarray(o["affected_input_cell_rows"],np.int64) for o in dop]))
 irows=np.unique(np.concatenate([np.asarray(o["source_cell_rows"],np.int64) for o in iops]))
 need(not np.intersect1d(ddrop,idrop).size and not np.intersect1d(ddrop,ikeep).size and not np.intersect1d(dkeep,idrop).size,"D500 and I528 endpoint maps are not disjoint")
 need(not np.intersect1d(drows,irows).size,"D500/I528 source cell stars overlap")
 # Compose both quotients into comp0's point prefix; its appended fan nodes remain identity.
 node=np.arange(len(cp),dtype=np.int64);node[:nbase]=dnode
 need(np.array_equal(node[idrop],idrop) and np.array_equal(node[ikeep],ikeep),"I528 endpoints conflict with D500 point map")
 node[idrop]=ikeep
 need(not np.intersect1d(ddrop,np.asarray(pr["protected_volume_point_rows"],np.int64)).size,"D500 drops a comp0/comp1 protected point")
 need(not np.intersect1d(idrop,np.asarray(pr["protected_volume_point_rows"],np.int64)).size,"I528 drops a comp0/comp1 protected point")
 need(np.array_equal(node[np.asarray(pr["protected_volume_point_rows"],np.int64)],np.asarray(pr["protected_volume_point_rows"],np.int64)),"combined map moves a protected point")
 # Exact material/corner key row mapping into the comp0 branch.
 bk=keys(bt,bm);ck=keys(ct,cm);order=np.argsort(ck,kind="stable");sk=ck[order]
 def map_rows_to_comp(rows,label):
  q=bk[np.asarray(rows,np.int64)];pos=np.searchsorted(sk,q);hit=pos<len(sk);hit[hit]&=(sk[pos[hit]]==q[hit])
  need(np.all(hit),label+" source cell keys missing from comp0")
  out=order[pos];need(len(np.unique(out))==len(out),label+" source cell key maps collide")
  return out,q
 dcomp,dk=map_rows_to_comp(drows,"D500");icomp,ik=map_rows_to_comp(irows,"I528")
 need(not np.intersect1d(dcomp,icomp).size,"D500/I528 stars overlap in comp0 rows")
 pkeys=np.asarray(pr["protected_tet_material_corner_keys"])
 need(not np.intersect1d(dk,pkeys).size and not np.intersect1d(ik,pkeys).size,"D500/I528 source stars hit comp0/comp1 protected keys")
 # Every and only operation-star cell in the comp0 mesh is incident to one of the retired endpoints.
 all_drops=np.concatenate((ddrop,idrop));need(len(np.unique(all_drops))==len(all_drops),"combined retired point rows collide")
 incident=np.flatnonzero(np.isin(ct,all_drops).any(axis=1));allcomp=np.concatenate((dcomp,icomp))
 need(np.array_equal(np.sort(incident),np.sort(allcomp)),"comp0 incident cells differ from authenticated D500/I528 full stars")
 mapped=node[ct];sortmapped=np.sort(mapped,axis=1);deg=np.any(sortmapped[:,1:]==sortmapped[:,:-1],axis=1)
 keep=np.flatnonzero(~deg);remove=np.flatnonzero(deg);outt=mapped[keep];outm=cm[keep]
 expected_removed=int(dr["removed_cell_count"])+sum(int(o["removed_cells"]) for o in iops)
 need(len(remove)==expected_removed,"composed D500/I528 removed-cell count differs")
 need(np.all(outm==16),"material labels changed")
 outkeys=keys(outt,outm);need(len(np.unique(outkeys))==len(outkeys),"duplicate tetrahedra after composition")
 # Reconcile transformed target: I528's typed target (397), then D500's endpoint map.
 itarget=np.asarray(t["desired_boundary_faces"],np.int64)
 icandtarget=np.asarray(i["desired_boundary_faces"],np.int64)
 need(np.array_equal(chain(icandtarget)[0],chain(itarget)[0]) and np.array_equal(chain(icandtarget)[1],chain(itarget)[1]),"typed target397 differs from I528 volume candidate target")
 itg=np.asarray(t["desired_boundary_source_group"],np.int8);itsr=np.asarray(t["desired_boundary_source_rows"],np.int64);itf=np.asarray(t["desired_boundary_source_face_rows"],np.int64)
 need(len(itarget)==len(itg)==len(itsr)==len(itf),"typed I528 target ancestry row counts differ")
 # Check I528 target is exactly the base target under its own quotient.
 ibt=np.asarray(b["desired_boundary_faces"],np.int64);ibg=np.asarray(b["desired_boundary_source_group"],np.int8);ibs=np.asarray(b["desired_boundary_source_rows"],np.int64);ibf=np.asarray(b["desired_boundary_source_face_rows"],np.int64)
 imt=imap[ibt];ims=np.sort(imt,axis=1);ideg=np.any(ims[:,1:]==ims[:,:-1],axis=1);iref=imt[~ideg]
 need(np.array_equal(chain(iref)[0],chain(itarget)[0]) and np.array_equal(chain(iref)[1],chain(itarget)[1]),"I528 typed target is not base target under I quotient")
 # Check D500 target against base target under D500 quotient before composing.
 dmt=dnode[ibt];dms=np.sort(dmt,axis=1);ddeg=np.any(dms[:,1:]==dms[:,:-1],axis=1);dref=dmt[~ddeg]
 dtarg=np.asarray(d["proposed_current_desired_boundary_faces"],np.int64)
 need(np.array_equal(chain(dref)[0],chain(dtarg)[0]) and np.array_equal(chain(dref)[1],chain(dtarg)[1]),"D500 transferred target is not base target under D quotient")
 tm=node[itarget];tms=np.sort(tm,axis=1);tdeg=np.any(tms[:,1:]==tms[:,:-1],axis=1)
 target=tm[~tdeg];tg=itg[~tdeg];ts=itsr[~tdeg];tf=itf[~tdeg]
 closed(target,"D500+I528 target")
 bo=boundary(outt)
 need(np.array_equal(chain(bo)[0],chain(target)[0]) and np.array_equal(chain(bo)[1],chain(target)[1]),"combined tet boundary differs from combined typed target")
 # Protect all walls under the combined quotient.
 wf=node[np.asarray(t["fat_current_target_rebind_wall_faces_volume_rows"],np.int64)]
 wsort=np.sort(wf,axis=1);wdeg=np.any(wsort[:,1:]==wsort[:,:-1],axis=1)
 need(not np.any(wdeg),"combined quotient collapses a wall face")
 # Recompute exact F32 lattice determinants for all input/output cells.
 units=[tuple(unit(q) for q in row) for row in cp]
 for lo in range(0,len(ct),10000):
  for j,tet in enumerate(ct[lo:lo+10000]):
   if det6(tet,units)<=0: raise ValueError(f"comp0 source exact nonpositive cell row {lo+j}")
 for row,tet in enumerate(outt):
  if det6(tet,units)<=0: raise ValueError(f"combined output exact nonpositive cell row {row}")
 # Exact local volume change over the authenticated union of source stars.
 starrows=np.unique(np.concatenate((drows,irows))); compstar=np.concatenate((dcomp,icomp))
 # Use the key-derived map and ensure one current comp row per base source star.
 bybase={int(r):int(v) for r,v in zip(drows,dcomp)}
 bybase.update({int(r):int(v) for r,v in zip(irows,icomp)})
 starcomp=np.asarray([bybase[int(r)] for r in starrows],np.int64)
 comp_to_out=np.full(len(ct),-1,np.int64);comp_to_out[keep]=np.arange(len(keep),dtype=np.int64)
 oldsum=newsum=0;all_delta=[]
 for r,cr in zip(starrows,starcomp):
  old=det6(ct[cr],units);need(old>0,"old operation-star cell exact sign failed");oldsum+=old
  mappedtet=node[ct[cr]]
  if len(set(map(int,mappedtet)))<4: delta=-old
  else:
   new=det6(mappedtet,units);need(new>0,"new operation-star cell exact sign failed");delta=new-old;newsum+=new
  all_delta.append(delta)
 exact_delta=sum(all_delta)
 need(newsum-oldsum==exact_delta,"exact local delta internal mismatch")
 ddelta=int(dr["sequential_exact_det6_delta_sum_integer"])
 # Compute I-only exact delta on source base star to independently reconcile its transfer.
 i_delta=0
 for r in irows:
  old=det6(bt[int(r)],units);newtet=imap[bt[int(r)]]
  new=0 if len(set(map(int,newtet)))<4 else det6(newtet,units)
  i_delta+=new-old
 iopdelta=[]
 for op in iops:
  od=0
  for r in op["source_cell_rows"]:
   r=int(r);old=det6(bt[r],units);nt=imap[bt[r]];nv=0 if len(set(map(int,nt)))<4 else det6(nt,units);od+=nv-old
  iopdelta.append(od)
 need(sum(iopdelta)==i_delta,"I528 per-operation exact deltas do not sum")
 need(exact_delta==ddelta+i_delta,"combined exact volume delta does not equal independent D and I deltas")
 # Quality census and no old-good regressions on comp0's exact input mesh.
 oldbad,q0,v0=quality(cp,ct);newbad,q1,v1=quality(cp,outt)
 regress=int(np.count_nonzero((~oldbad[keep])&newbad));need(regress==0,"combined quotient creates a native-bad cell from old-good input")
 # Build current comp0 row maps and preserve the source-provenance arrays as pre-composition evidence.
 z={k:np.array(v,copy=True) for k,v in c.items()}
 for k in list(z):
  if any(x in k for x in ("output_cell_to_parent","output_cell_to_source","replacement_output_cell_rows","replacement_output_tetrahedra_global_node_rows")):
   z["inherited_54138_"+k]=z.pop(k)
 comp_to_out=np.full(len(ct),-1,np.int64);comp_to_out[keep]=np.arange(len(keep),dtype=np.int64)
 repl=np.asarray(c["comp0_replacement_output_cell_rows"],np.int64);mapped_repl=comp_to_out[repl]
 need(np.all(mapped_repl>=0),"combined quotient overlaps comp0 replacement cells")
 need(np.array_equal(outt[mapped_repl],c["comp0_replacement_output_tetrahedra_global_node_rows"]),"comp0 replacement tets changed under disjoint quotient")
 z["comp0_replacement_output_cell_rows"]=mapped_repl
 # Retain full per-operation star maps in the immediate 54138 input row domain.
 off=[0];star_comp=[]
 for op in dop:
  q=np.asarray(op["affected_input_cell_rows"],np.int64);rr,_=map_rows_to_comp(q,"D500 operation");star_comp.extend(map(int,rr));off.append(len(star_comp))
 for op in iops:
  q=np.asarray(op["source_cell_rows"],np.int64);rr,_=map_rows_to_comp(q,"I528 operation");star_comp.extend(map(int,rr));off.append(len(star_comp))
 z["points_world_m"]=cp;z["points_world_m_f32"]=cp;z["tetrahedra"]=outt;z["material_indices"]=outm;z["boundary_faces"]=bo
 z["retained_input_cell_rows"]=keep
 cm2=np.full(len(ct),-1,np.int64);cm2[keep]=np.arange(len(keep),dtype=np.int64)
 z["input_cell_to_output_cell"]=cm2;z["output_cell_to_input_cell"]=keep
 z["input_node_to_output_node"]=node;z["output_node_to_input_node"]=np.arange(len(cp),dtype=np.int64)
 z["mapped_current_boundary_faces"]=bo
 z["desired_boundary_faces"]=target;z["desired_boundary_source_group"]=tg;z["desired_boundary_source_rows"]=ts;z["desired_boundary_source_face_rows"]=tf
 z["proposed_current_desired_boundary_faces"]=target;z["proposed_current_desired_boundary_source_group"]=tg;z["proposed_current_desired_boundary_source_rows"]=ts;z["proposed_current_desired_boundary_source_face_rows"]=tf
 z["operation_cell_star_offsets"]=np.asarray(off,np.int64);z["operation_cell_star_input_rows"]=np.asarray(star_comp,np.int64)
 z["operation_surface_ledger_indices"]=np.arange(len(dop)+len(iops),dtype=np.int64)
 z["operation_retired_volume_point_rows"]=all_drops;z["operation_kept_volume_point_rows"]=np.concatenate((dkeep,ikeep))
 z["operation_exact_det6_delta_integer"]=np.asarray([str(o["exact_det6_delta_integer"]) for o in dop]+[str(x) for x in iopdelta],dtype="U256")
 z["D500_I528_base_54381_input_cell_to_comp0_cell"]=np.full(len(bt),-1,np.int64)
 # Fill exact base->comp map only where a source cell survives on the comp0 branch.
 basepos=np.searchsorted(sk,bk);basehit=basepos<len(sk);basehit[basehit]&=(sk[basepos[basehit]]==bk[basehit])
 b2c=np.full(len(bt),-1,np.int64);b2c[basehit]=order[basepos[basehit]]
 z["D500_I528_base_54381_input_cell_to_comp0_cell"]=b2c
 base_to_out=np.full(len(bt),-1,np.int64);good=b2c>=0;base_to_out[good]=comp_to_out[b2c[good]]
 z["D500_I528_base_54381_input_cell_to_composed_cell"]=base_to_out
 z["D500_I528_comp0_cell_to_composed_cell"]=comp_to_out
 z["D500_I528_base_54381_input_node_to_composed_node"]=node[:nbase]
 z["D500_I528_comp0_input_node_to_composed_node"]=node
 z["D500_I528_D_star_base_cell_rows"]=drows;z["D500_I528_I_star_base_cell_rows"]=irows
 z["D500_I528_D_star_comp0_cell_rows"]=dcomp;z["D500_I528_I_star_comp0_cell_rows"]=icomp
 z["D500_I528_input_node_drop_rows"]=all_drops
 z["D500_I528_input_node_keep_rows"]=np.concatenate((dkeep,ikeep))
 z["D500_I528_exact_source_cell_key_composition"]=np.asarray([True])
 z["D500_I528_disjoint_D_I_stars"]=np.asarray([True])
 z["D500_I528_comp0_comp1_fullstar_disjoint"]=np.asarray([True])
 z["D500_I528_typed_D_697_binding_status"]=np.asarray(["PENDING"],dtype="U16")
 # Rebind the target geometry arrays from I528's independently typed target, then apply D500's quotient.
 z["fat_current_target_rebind_schema"]=t["fat_current_target_rebind_schema"]
 for name in ("fat_current_target_rebind_D_point_to_volume_rows","fat_current_target_rebind_I_point_to_volume_rows"):
  z[name]=node[np.asarray(t[name],np.int64)]
 z["fat_current_target_rebind_wall_faces_volume_rows"]=wf
 z["fat_current_target_rebind_wall_faces_collar_point_rows"]=t["fat_current_target_rebind_wall_faces_collar_point_rows"]
 z["fat_current_target_rebind_wall_face_points_world_m_f32"]=cp[wf]
 for name in ("fat_current_target_rebind_source_asset_sha256_json","fat_current_target_rebind_typed_I_candidate_path","fat_current_target_rebind_typed_I_candidate_sha256","fat_current_target_rebind_typed_O_candidate_path","fat_current_target_rebind_typed_O_candidate_sha256"):
  if name in t:z[name]=t[name]
 for name in ("fat_current_target_rebind_typed_D_candidate_path","fat_current_target_rebind_typed_D_candidate_sha256","fat_current_target_rebind_typed_D_697_receipt_path","fat_current_target_rebind_typed_D_697_receipt_sha256","fat_current_target_rebind_typed_D_lineage_correction_receipt_path","fat_current_target_rebind_typed_D_lineage_correction_receipt_sha256"):
  if name in z:z[name]=np.asarray([""],dtype="U512")
 z["fat_current_target_rebind_metadata_mode"]=np.asarray(["D500+I528 exact volume composition; new D typed handoff/697 pending"],dtype="U160")
 # Provenance pins and array-hash sidecar.
 z["D500_input_path"]=np.asarray([str(BASE)],dtype="U512");z["D500_input_sha256"]=np.asarray([pins[str(BASE)]],dtype="U64")
 z["D500_transfer_path"]=np.asarray([str(DOUT)],dtype="U512");z["D500_transfer_sha256"]=np.asarray([pins[str(DOUT)]],dtype="U64")
 z["I528_input_path"]=np.asarray([str(IINPUT)],dtype="U512");z["I528_input_sha256"]=np.asarray([pins[str(IINPUT)]],dtype="U64")
 z["I528_typed_target_path"]=np.asarray([str(ITARGET)],dtype="U512");z["I528_typed_target_sha256"]=np.asarray([pins[str(ITARGET)]],dtype="U64")
 z["D500_I528_comp0_parent_path"]=np.asarray([str(COMP)],dtype="U512");z["D500_I528_comp0_parent_sha256"]=np.asarray([pins[str(COMP)]],dtype="U64")
 z["D500_I528_protection007_path"]=np.asarray([str(PROT)],dtype="U512");z["D500_I528_protection007_sha256"]=np.asarray([pins[str(PROT)]],dtype="U64")
 np.savez_compressed(CANDIDATE,**z);outsha=sha(CANDIDATE)
 report={
  "schema":"h706.d500-i528-comp0-exact-key-composition.v1","status":"PASS_D500_I528_COMPOSED_ON_COMP0_EXACT_KEYS",
  "candidate":{"path":str(CANDIDATE),"sha256":outsha},
  "inputs":{str(p):{"sha256":pins[str(p)]} for p in (BASE,DOUT,COMP,PROT,IINPUT,IREPORT,ITARGET,ITYPED,ITYPEDRECEIPT)},
  "operations":{"D500":len(dop),"I528":len(iops),"D500_I528_source_cell_stars_disjoint":True,"D500_I528_retired_and_kept_node_sets_disjoint":True,"all_3945_I528_source_keys_present_once_in_comp0":True,"D500_affected_keys_present_once_in_comp0":True,"protected_comp0_comp1_keys_intersections":0,"protected_point_retirements":0},
  "geometry":{"input_comp0_cells":len(ct),"output_cells":len(outt),"removed_cells":len(remove),"input_points":len(cp),"points_bitwise_unchanged":True,"all_exact_f32_det6_positive":True,"target_faces":len(target),"volume_boundary_equals_target":True,"target_missing_oriented_records":0,"target_extra_oriented_records":0,"closed_oriented_target":True,"wall_faces_preserved":True},
  "exact_volume_delta":{"det6_integer":str(exact_delta),"det6_denominator":str(1<<447),"volume_m3":str(exact_delta/(6*(1<<447))),"equals_D500_plus_I528":True},
  "quality":{"native_bad_before":int(oldbad.sum()),"native_bad_after":int(newbad.sum()),"gain":int(oldbad.sum()-newbad.sum()),"old_good_regressions":regress,"minimum_normalized_det_before":q0,"minimum_normalized_det_after":q1},
  "target_source":{"I_target_path":str(ITARGET),"I_target_sha256":pins[str(ITARGET)],"D500_surface_candidate_path":str(DSURF),"D500_surface_candidate_sha256":pins[str(DSURF)],"fresh_D_697_binding":"PENDING"},
  "limitations":["Source-only geometry composition; no mass mapping or physical admission.","Fresh typed D surface lineage and actual 697 load remain pending.","Final combined changed-face audit against current O/I/D, both walls and all185 remains pending.","No native installation or native steps were run."],
  "physical_admission":False,"mass_mapping":"NOT_RUN","native_steps":0,"elapsed_seconds":time.monotonic()-start
 }
 (OUTDIR/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"candidate":str(CANDIDATE),"sha256":outsha,"cells":[len(ct),len(outt)],"removed":len(remove),"quality":[int(oldbad.sum()),int(newbad.sum())],"exact_delta":str(exact_delta),"elapsed_seconds":round(report["elapsed_seconds"],3)},sort_keys=True))
try: main()
except Exception as e:
 OUTDIR.mkdir(parents=True,exist_ok=True);p=OUTDIR/"failure.json";p.write_text(json.dumps({"schema":"h706.d500-i528-comp0-exact-key-composition.v1","status":"FAIL_D500_I528_COMPOSED_ON_COMP0_EXACT_KEYS","error":repr(e),"traceback":traceback.format_exc()},sort_keys=True,indent=2)+"\n");print(json.dumps({"status":"FAIL","error":repr(e),"report":str(p),"sha256":sha(p)}));raise

