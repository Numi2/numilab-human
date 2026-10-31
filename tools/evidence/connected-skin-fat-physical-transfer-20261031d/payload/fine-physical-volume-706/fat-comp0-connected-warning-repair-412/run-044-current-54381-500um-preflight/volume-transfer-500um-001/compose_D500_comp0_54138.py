#!/usr/bin/env python3
"""Compose the exact D500 endpoint quotient onto the protected comp0 fan branch."""
from pathlib import Path
import hashlib,json,time,traceback
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight"
BASE=H/"fat-comp0-connected-warning-repair-412/run-042-current-54381-normalized-rebind/fat-current-typed-target-rebound-54381-lineage-corrected.npz"
DOUT=R/"volume-transfer-500um-001/run001/fat-D500-short-edge-quotient-volume-proposal.npz"
DREPORT=R/"volume-transfer-500um-001/run001/report.json"
DOPMAP=R/"volume-transfer-500um-001/run001/operation-volume-star-map.jsonl"
SURF=R/"surface-proposal-500um/candidate-D-short-edge-collapse-raw.npz"
SREPORT=R/"surface-proposal-500um/report.json"
COMP=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-007-kernel-fan-composed/fat-comp0-kernel-fan-composed-volume-proposal.npz"
COMPREPORT=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-007-kernel-fan-composed/report.json"
PROT=H/"fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/run-038-kernel-fans-297/protection-007/comp0-comp1-protected-stars.npz"
OUTDIR=R/"volume-transfer-500um-001/compose-comp0-54138-001"
CANDIDATE=OUTDIR/"fat-D500-comp0-kernel-fan-composed-volume-proposal.npz"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def need(x,msg):
 if not x:raise ValueError(msg)
def chain(f):
 a=np.asarray(f,np.int64).reshape((-1,3));ok=np.all(np.diff(np.sort(a,axis=1),axis=1)!=0,axis=1);a=a[ok]
 if not len(a):return np.empty(0,np.int64),np.empty(0,np.int64)
 s=np.sort(a,axis=1);inv=(a[:,0]>a[:,1]).astype(np.int8)+(a[:,0]>a[:,2]).astype(np.int8)+(a[:,1]>a[:,2]).astype(np.int8);sg=(1-2*(inv&1)).astype(np.int64);k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];su=np.add.reduceat(sg,st);kk=k[st];keep=su!=0;return kk[keep],su[keep]
def boundary(t):
 F=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64);f=np.asarray(t,np.int64)[:,F].reshape((-1,3));s=np.sort(f,axis=1);inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8);sg=(1-2*(inv&1)).astype(np.int8);k=(s[:,0]*(1<<20)+s[:,1])*(1<<20)+s[:,2];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1];ct=np.diff(np.r_[st,len(k)]);su=np.add.reduceat(sg,st)
 if np.any(ct>2) or np.any((ct==2)&(su!=0)):raise ValueError("nonmanifold or inconsistent tet face")
 return f[o[st[ct==1]]]
def closed(f,label):
 a=np.asarray(f,np.int64);e=np.concatenate((a[:,[0,1]],a[:,[1,2]],a[:,[2,0]]));sg=np.where(e[:,0]<e[:,1],1,-1).astype(np.int8);s=np.sort(e,axis=1);k=s[:,0]*(1<<20)+s[:,1];o=np.argsort(k,kind="stable");k=k[o];sg=sg[o];st=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1]
 if np.any(np.add.reduceat(sg,st)):raise ValueError(label+" not closed")
def makekeys(t,m):
 dt=np.dtype([("mat","<i4"),("a","<i8"),("b","<i8"),("c","<i8"),("d","<i8")]);s=np.sort(np.asarray(t,np.int64),axis=1);a=np.empty(len(s),dt);a["mat"]=np.asarray(m,np.int32);a["a"]=s[:,0];a["b"]=s[:,1];a["c"]=s[:,2];a["d"]=s[:,3];return a
def quality(p,t):
 bad=[];qmin=float("inf");vmin=float("inf")
 for lo in range(0,len(t),50000):
  x=np.asarray(p[np.asarray(t[lo:lo+50000],np.int64)],np.float64);a=x[:,1]-x[:,0];b=x[:,2]-x[:,0];c=x[:,3]-x[:,0];d=np.einsum("ij,ij->i",a,np.cross(b,c));es=[]
  for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
   e=x[:,i]-x[:,j];es.append(np.einsum("ij,ij->i",e,e))
  mx=np.maximum.reduce(es);q=np.abs(d)/np.power(mx,1.5);v=np.abs(d)/6.;bad.append((v<=1e-18)|(q<1e-5));qmin=min(qmin,float(q.min()));vmin=min(vmin,float(v.min()))
 return np.concatenate(bad),qmin,vmin
def unit(v):
 b=int(np.asarray(v,dtype=np.float32).view(np.uint32));sign=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255:raise ValueError("nonfinite F32 point")
 return sign*f if e==0 else sign*((1<<23)|f)<<(e-1)
def det6(t,u):
 p0,p1,p2,p3=(u[int(i)] for i in t);a=[p1[j]-p0[j] for j in range(3)];b=[p2[j]-p0[j] for j in range(3)];c=[p3[j]-p0[j] for j in range(3)]
 return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def main():
 OUTDIR.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 b=load(BASE);d=load(DOUT);c=load(COMP);pr=load(PROT);dr=json.loads(DREPORT.read_text());sr=json.loads(SREPORT.read_text());cr=json.loads(COMPREPORT.read_text())
 pins={str(x):sha(x) for x in (BASE,DOUT,DREPORT,DOPMAP,SURF,SREPORT,COMP,COMPREPORT,PROT)}
 need(pins[str(BASE)]=="5aa02086aac4b20e879332cd9d5ba64da636dd8a520de54e7b96b00225c7fe42","base pin mismatch")
 need(pins[str(DOUT)]=="35a5545ed4d3bbe40499df0b143456683fb8f0739372e2a53b8a3227e4474b24","D output pin mismatch")
 need(pins[str(COMP)]=="54138c6a3cbc62ff37a0bf9d86257896587304a493f9e05f9c9ec02fc3dc80fb","comp0 output pin mismatch")
 bp=np.asarray(b["points_world_m_f32"],np.float32);bt=np.asarray(b["tetrahedra"],np.int64);bm=np.asarray(b["material_indices"],np.int32)
 dp=np.asarray(d["points_world_m_f32"],np.float32);dt=np.asarray(d["tetrahedra"],np.int64);dm=np.asarray(d["material_indices"],np.int32);node=np.asarray(d["input_node_to_output_node"],np.int64)
 cp=np.asarray(c["points_world_m_f32"],np.float32);ct=np.asarray(c["tetrahedra"],np.int64);cm=np.asarray(c["material_indices"],np.int32)
 need(np.array_equal(bp,dp),"D volume transfer changed point coordinates")
 need(len(cp)>=len(bp) and np.array_equal(bp,cp[:len(bp)]),"comp0 point prefix differs from D500 source")
 need(np.array_equal(c["points_world_m"],cp),"comp0 point views disagree")
 need(len(node)==len(bp) and np.all((node>=0)&(node<len(cp))),"D point map domain/range invalid")
 extended=np.arange(len(cp),dtype=np.int64);extended[:len(node)]=node
 ops=[json.loads(z) for z in DOPMAP.read_text().splitlines() if z.strip()]
 need(len(ops)==1277 and int(dr["transferred_operation_count"])==1277,"D op count mismatch")
 drops=np.asarray([z["drop_volume_row"] for z in ops],np.int64);keeps=np.asarray([z["keep_volume_row"] for z in ops],np.int64)
 need(not np.intersect1d(drops,np.asarray(pr["protected_volume_point_rows"],np.int64)).size,"D drop point intersects comp0/comp1 protected point set")
 need(np.array_equal(extended[np.asarray(pr["protected_volume_point_rows"],np.int64)],np.asarray(pr["protected_volume_point_rows"],np.int64)),"D quotient remaps protected comp0/comp1 point")
 # Match all 54381 D operation cells into the comp0 candidate by exact material/corner keys.
 affected=np.unique(np.concatenate([np.asarray(z["affected_input_cell_rows"],np.int64) for z in ops]))
 basekeys=makekeys(bt,bm);compkeys=makekeys(ct,cm);order=np.argsort(compkeys,kind="stable");sorted_comp=compkeys[order]
 ak=basekeys[affected];pos=np.searchsorted(sorted_comp,ak);hit=pos<len(sorted_comp);hit[hit]&=sorted_comp[pos[hit]]==ak[hit]
 need(np.all(hit),"D source affected cell keys absent from comp0 branch")
 comp_rows=order[pos]
 need(len(np.unique(comp_rows))==len(comp_rows),"D affected cell keys collide in comp0 branch")
 protkeys=np.asarray(pr["protected_tet_material_corner_keys"])
 need(not np.intersect1d(ak,protkeys).size,"D affected source cells overlap protected comp0/comp1 cells")
 incident=np.flatnonzero(np.isin(ct,drops).any(axis=1))
 need(np.array_equal(np.sort(incident),np.sort(comp_rows)),"comp0 candidate has new or missing D-drop incident cells")
 # Confirm the candidate's pre-quotient target is the exact 54381 target chain.
 ctarget=np.asarray(c["proposed_current_desired_boundary_faces"],np.int64)
 btarget=np.asarray(b["desired_boundary_faces"],np.int64)
 need(np.array_equal(chain(ctarget)[0],chain(btarget)[0]) and np.array_equal(chain(ctarget)[1],chain(btarget)[1]),"comp0 and 54381 targets differ")
 mapped=extended[ct];sm=np.sort(mapped,axis=1);deg=np.any(sm[:,1:]==sm[:,:-1],axis=1)
 keep=np.flatnonzero(~deg);remove=np.flatnonzero(deg);outt=mapped[keep];outm=cm[keep]
 need(len(remove)==int(dr["removed_cell_count"])==4065,"D composed deleted cell count differs")
 outkeys=makekeys(outt,outm);need(len(np.unique(outkeys))==len(outkeys),"D composition creates duplicate tetrahedra")
 need(np.all(outm==16),"D composition changed material")
 bo=boundary(outt);btmapped=extended[np.asarray(c["boundary_faces"],np.int64)];bs=np.sort(btmapped,axis=1);bd=np.any(bs[:,1:]==bs[:,:-1],axis=1);btmapped=btmapped[~bd]
 target=np.asarray(d["proposed_current_desired_boundary_faces"],np.int64);tg=np.asarray(d["proposed_current_desired_boundary_source_group"],np.int8);ts=np.asarray(d["proposed_current_desired_boundary_source_rows"],np.int64);tf=np.asarray(d["proposed_current_desired_boundary_source_face_rows"],np.int64)
 need(np.array_equal(chain(bo)[0],chain(btmapped)[0]) and np.array_equal(chain(bo)[1],chain(btmapped)[1]),"composed boundary not the quotient of comp0 boundary")
 need(np.array_equal(chain(bo)[0],chain(target)[0]) and np.array_equal(chain(bo)[1],chain(target)[1]),"composed volume boundary differs from D500 current target")
 need(np.array_equal(chain(btmapped)[0],chain(target)[0]) and np.array_equal(chain(btmapped)[1],chain(target)[1]),"mapped comp0 target differs from D500 target")
 closed(bo,"composed volume boundary");closed(target,"D500 target")
 # Exact full-source and replacement signs, with an exact local det6 delta replay.
 units=[tuple(unit(q) for q in row) for row in cp]
 min_source=None
 for lo in range(0,len(ct),10000):
  for j,tet in enumerate(ct[lo:lo+10000]):
   q=det6(tet,units)
   if q<=0:raise ValueError(f"comp0 input exact nonpositive tet row {lo+j}")
   if min_source is None or q<min_source:min_source=q
 olds=np.asarray(ct[comp_rows],np.int64);news=extended[olds];oldsum=0;newsum=0;newcount=0;delcount=0
 for a,z in zip(olds,news):
  x=det6(a,units);need(x>0,"D affected comp0 source tet not exact-positive");oldsum+=x
  if len(set(map(int,z)))<4:delcount+=1;continue
  y=det6(z,units);need(y>0,"D replacement comp0 tet exact nonpositive");newsum+=y;newcount+=1
 delta=newsum-oldsum
 need(str(delta)==str(dr["sequential_exact_det6_delta_sum_integer"]),"composed exact volume delta differs from D-only transfer")
 need(delcount==len(remove),"D affected star deleted cell count mismatch")
 oldbad,q0,v0=quality(cp,ct);newbad,q1,v1=quality(cp,outt)
 need(int(oldbad.sum())==int(cr["whole_candidate"]["native_bad_cell_count"] if "native_bad_cell_count" in cr["whole_candidate"] else int(oldbad.sum())),"comp0 native quality baseline mismatch")
 reg=int(np.count_nonzero((~oldbad[keep])&newbad));need(reg==0,"D composition regresses old-good comp0 cell")
 # Independent base-to-comp0 and parent-to-output maps for disjoint exact-key composition.
 basepos=np.searchsorted(sorted_comp,basekeys);basehit=basepos<len(sorted_comp);basehit[basehit]&=sorted_comp[basepos[basehit]]==basekeys[basehit]
 base_to_comp=np.full(len(bt),-1,np.int64);base_to_comp[basehit]=order[basepos[basehit]]
 comp_to_out=np.full(len(ct),-1,np.int64);comp_to_out[keep]=np.arange(len(keep),dtype=np.int64)
 base_to_out=np.full(len(bt),-1,np.int64);good=base_to_comp>=0;base_to_out[good]=comp_to_out[base_to_comp[good]]
 need(np.all(base_to_comp[affected]>=0),"one or more D stars cannot be composed by stable source cell key")
 # Update inherited comp0 row-domain arrays without losing the original 40c90 lineage maps.
 z={k:np.array(v,copy=True) for k,v in c.items()}
 for k in ("input_cell_to_output_cell","output_cell_to_input_cell","retained_input_cell_rows","input_node_to_output_node","output_node_to_input_node","operation_cell_star_offsets","operation_cell_star_input_rows","operation_surface_ledger_indices","operation_retired_volume_point_rows","operation_kept_volume_point_rows","operation_exact_det6_delta_integer"):
  if k in z:z["inherited_54138_"+k]=z[k]
 for k in ("comp0_output_cell_to_parent_D480_rows","comp0_output_cell_to_source480_rows"):
  if k in z and len(z[k])==len(ct):z[k]=z[k][keep]
 if "comp0_replacement_output_cell_rows" in z:
  rr=np.asarray(z["comp0_replacement_output_cell_rows"],np.int64);mappedrr=comp_to_out[rr]
  need(np.all(mappedrr>=0),"D quotient overlaps comp0 replacement rows")
  z["comp0_replacement_output_cell_rows"]=mappedrr
  if "comp0_replacement_output_tetrahedra_global_node_rows" in z:
   need(np.array_equal(outt[mappedrr],z["comp0_replacement_output_tetrahedra_global_node_rows"]),"comp0 replacement tet lineage changed")
 # Compose the D500 operation-star map into the current 54138 cell row domain.
 off=[0];starrows=[]
 for entry in ops:
  srcrows=np.asarray(entry["affected_input_cell_rows"],np.int64);mappedrows=base_to_comp[srcrows]
  need(np.all(mappedrows>=0),"D operation star lacks a comp0 current-cell key")
  starrows.extend(map(int,mappedrows));off.append(len(starrows))
 z["points_world_m"]=cp;z["points_world_m_f32"]=cp;z["tetrahedra"]=outt;z["material_indices"]=outm;z["boundary_faces"]=bo
 z["retained_input_cell_rows"]=keep
 ic=np.full(len(ct),-1,np.int64);ic[keep]=np.arange(len(keep),dtype=np.int64)
 z["input_cell_to_output_cell"]=ic;z["output_cell_to_input_cell"]=keep
 z["input_node_to_output_node"]=extended;z["output_node_to_input_node"]=np.arange(len(cp),dtype=np.int64)
 z["mapped_current_boundary_faces"]=bo
 z["desired_boundary_faces"]=target;z["desired_boundary_source_group"]=tg;z["desired_boundary_source_rows"]=ts;z["desired_boundary_source_face_rows"]=tf
 z["proposed_current_desired_boundary_faces"]=target;z["proposed_current_desired_boundary_source_group"]=tg;z["proposed_current_desired_boundary_source_rows"]=ts;z["proposed_current_desired_boundary_source_face_rows"]=tf
 z["operation_cell_star_offsets"]=np.asarray(off,np.int64);z["operation_cell_star_input_rows"]=np.asarray(starrows,np.int64)
 z["operation_surface_ledger_indices"]=np.arange(len(ops),dtype=np.int64);z["operation_retired_volume_point_rows"]=drops;z["operation_kept_volume_point_rows"]=keeps
 z["operation_exact_det6_delta_integer"]=np.asarray([str(e["exact_det6_delta_integer"]) for e in ops],dtype="U256")
 z["D500_base_54381_input_cell_to_comp0_cell"]=base_to_comp;z["D500_base_54381_input_cell_to_composed_cell"]=base_to_out
 z["D500_comp0_cell_to_composed_cell"]=comp_to_out;z["D500_base_54381_input_node_to_composed_node"]=node
 z["D500_star_base_cell_rows"]=affected;z["D500_star_comp0_cell_rows"]=comp_rows
 z["D500_source_volume_path"]=np.asarray([str(BASE)],dtype="U512");z["D500_source_volume_sha256"]=np.asarray([pins[str(BASE)]],dtype="U64")
 z["D500_volume_quotient_path"]=np.asarray([str(DOUT)],dtype="U512");z["D500_volume_quotient_sha256"]=np.asarray([pins[str(DOUT)]],dtype="U64")
 z["D500_surface_candidate_path"]=np.asarray([str(SURF)],dtype="U512");z["D500_surface_candidate_sha256"]=np.asarray([pins[str(SURF)]],dtype="U64")
 z["D500_comp0_parent_candidate_path"]=np.asarray([str(COMP)],dtype="U512");z["D500_comp0_parent_candidate_sha256"]=np.asarray([pins[str(COMP)]],dtype="U64")
 z["D500_protection007_path"]=np.asarray([str(PROT)],dtype="U512");z["D500_protection007_sha256"]=np.asarray([pins[str(PROT)]],dtype="U64")
 z["D500_exact_cell_key_composition"]=np.asarray([True]);z["D500_fullstar_comp0_comp1_disjoint"]=np.asarray([True]);z["D500_typed_697_binding_status"]=np.asarray(["PENDING"],dtype="U16")
 # Store current target rebind geometry maps; typed source pins will be replaced by the 697 handoff.
 z["fat_current_target_rebind_schema"]=b["fat_current_target_rebind_schema"]
 z["fat_current_target_rebind_D_point_to_volume_rows"]=extended[np.asarray(b["fat_current_target_rebind_D_point_to_volume_rows"],np.int64)]
 z["fat_current_target_rebind_I_point_to_volume_rows"]=np.asarray(b["fat_current_target_rebind_I_point_to_volume_rows"],np.int64)
 wf=np.asarray(b["fat_current_target_rebind_wall_faces_volume_rows"],np.int64);wfm=extended[wf]
 wsort=np.sort(wfm,axis=1);wdeg=np.any(wsort[:,1:]==wsort[:,:-1],axis=1);need(not np.any(wdeg),"D quotient collapses a protected wall triangle")
 z["fat_current_target_rebind_wall_faces_volume_rows"]=wfm
 z["fat_current_target_rebind_wall_faces_collar_point_rows"]=b["fat_current_target_rebind_wall_faces_collar_point_rows"]
 z["fat_current_target_rebind_wall_face_points_world_m_f32"]=cp[wfm]
 # Explicitly blank stale typed-D claims inherited from the 54381 metadata parent.
 for k in ("fat_current_target_rebind_typed_D_candidate_path","fat_current_target_rebind_typed_D_candidate_sha256","fat_current_target_rebind_typed_D_697_receipt_path","fat_current_target_rebind_typed_D_697_receipt_sha256","fat_current_target_rebind_typed_D_lineage_correction_receipt_path","fat_current_target_rebind_typed_D_lineage_correction_receipt_sha256"):
  if k in b:z[k]=np.asarray([""],dtype="U512")
 z["fat_current_target_rebind_metadata_mode"]=np.asarray(["D500 source-only composition; fresh typed D and 697 binding pending"],dtype="U128")
 np.savez_compressed(CANDIDATE,**z);outsha=sha(CANDIDATE)
 report={"schema":"h706.d500-comp0-exact-key-composition.v1","status":"PASS_D500_COMPOSED_ON_COMP0_EXACT_KEYS","input_54381":{"path":str(BASE),"sha256":pins[str(BASE)]},"d500_volume":{"path":str(DOUT),"sha256":pins[str(DOUT)],"report_sha256":pins[str(DREPORT)],"operation_map_sha256":pins[str(DOPMAP)]},"comp0_candidate":{"path":str(COMP),"sha256":pins[str(COMP)],"report_sha256":pins[str(COMPREPORT)]},"protection007":{"path":str(PROT),"sha256":pins[str(PROT)]},"candidate":{"path":str(CANDIDATE),"sha256":outsha},"input_comp0_cells":len(ct),"output_composed_cells":len(outt),"removed_D_cells":len(remove),"input_composed_points":len(cp),"source_54381_point_prefix_bitwise_identical":True,"D_affected_source_cell_rows":len(affected),"D_affected_keys_present_one_to_one_in_comp0":True,"D_affected_keys_intersect_comp0_protected_keys":0,"D_drop_points_in_comp0_protected_points":0,"D_keep_points_in_comp0_protected_points":int(len(np.intersect1d(keeps,np.asarray(pr["protected_volume_point_rows"],np.int64)))),"comp0_tets_incident_to_D_drops":len(incident),"all_incident_tets_are_authenticated_D_stars":True,"exact_det6_delta_integer":str(delta),"exact_det6_denominator":str(1<<447),"exact_volume_delta_m3":str(delta/(6*(1<<447))),"source_comp0_all_exact_positive_cells_recomputed":len(ct),"source_comp0_all_exact_positive":True,"output_exact_positive_for_changed_star":True,"native_bad_cells_before":int(oldbad.sum()),"native_bad_cells_after":int(newbad.sum()),"native_bad_cell_gain":int(oldbad.sum()-newbad.sum()),"old_good_cell_regressions":reg,"minimum_normalized_det_before":q0,"minimum_normalized_det_after":q1,"target_faces":len(target),"target_chain_matches_transferred_D500_target":True,"volume_boundary_equals_target":True,"target_missing_oriented_records":0,"target_extra_oriented_records":0,"comp0_cell_maps_reindexed":True,"D500_cell_maps_emitted":True,"typed_697_binding":"PENDING","mass_mapping":"NOT_RUN","native_steps":0,"physical_admission":False,"elapsed_seconds":time.monotonic()-start,"limitations":["This is a geometric D500 plus comp0 fan composition; it does not include paired-I528 operations.","A fresh typed D surface handoff/697 load and final joint current O/I/D/all185 audit remain required.","Source mass mapping, native installation and physical admission are not claimed."]}
 (OUTDIR/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"candidate":str(CANDIDATE),"sha256":outsha,"cells":[len(ct),len(outt)],"affected":len(affected),"removed":len(remove),"native_bad":[int(oldbad.sum()),int(newbad.sum())],"exact_delta":str(delta),"elapsed_seconds":round(report["elapsed_seconds"],3)},sort_keys=True))
try:main()
except Exception as e:
 OUTDIR.mkdir(parents=True,exist_ok=True);p=OUTDIR/"failure.json";p.write_text(json.dumps({"schema":"h706.d500-comp0-exact-key-composition.v1","status":"FAIL_D500_COMPOSED_ON_COMP0_EXACT_KEYS","error":repr(e),"traceback":traceback.format_exc()},sort_keys=True,indent=2)+"\n");print(json.dumps({"status":"FAIL","error":repr(e),"report":str(p),"sha256":sha(p)}));raise
