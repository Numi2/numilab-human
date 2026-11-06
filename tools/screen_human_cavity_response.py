#!/usr/bin/env python3
"""Preparation-only response audit for fixed-coordinate Human mesh proposals.

This compares the pinned existing skin/fat constitutive laws at a retained field;
it never advances simulation, supplies forces, changes material data, or admits a
physical step. The native owner remains the acceptance authority.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,re,time,subprocess
import numpy as np

def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def module(p,name,digest):
 if sha(p)!=digest:raise ValueError("owner SHA mismatch: "+str(p))
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def material(path,digest):
 if sha(path)!=digest:raise ValueError("material source mismatch")
 text=re.sub(r"//[^\n]*","",Path(path).read_text())
 compact=re.sub(r"\s+","",text)
 expected="energy=0.5*shear*(I1()-3-2*log(J()))+0.5*bulk*pow(log(J()),2);"
 if expected not in compact or "modelgeneric;" not in compact:raise ValueError("not the pinned eligible stateless energy law")
 values={}
 for key in ["shear","bulk"]:
  match=re.search(r"parameter\s+"+key+r"\s*:\s*MPa\s*=\s*([0-9.eE+-]+)",text)
  if not match:raise ValueError("missing MPa parameter")
  values[key]=float(match.group(1))*1e6
 return values

def response(points,cells,u,shear,bulk):
 """Double-precision reference evaluation of the pinned log-J energy only."""
 x=np.asarray(points,float)[cells];v=np.asarray(u,float)[cells]
 dm=np.swapaxes(x[:,1:]-x[:,:1],1,2);du=np.swapaxes(v[:,1:]-v[:,:1],1,2)
 inv=np.linalg.inv(dm);volume=np.linalg.det(dm)/6
 if not np.all(np.isfinite(volume)) or np.any(volume<=0):raise ValueError("invalid reference cells")
 h=du@inv;f=np.eye(3)+h
 tr=np.trace(h,axis1=1,axis2=2)
 quadratic=h[:,0,0]*h[:,1,1]+h[:,0,0]*h[:,2,2]+h[:,1,1]*h[:,2,2]-h[:,0,1]*h[:,1,0]-h[:,0,2]*h[:,2,0]-h[:,1,2]*h[:,2,1]
 jm=tr+quadratic+np.linalg.det(h)
 if not np.all(np.isfinite(jm)) or np.any(jm<=-1):raise ValueError("invalid deformed cells")
 logj=np.log1p(jm);fit=np.swapaxes(np.linalg.inv(f),1,2)
 # Equivalent form avoids subtracting F and F^-T near the identity.
 p=shear*(h+fit@np.swapaxes(h,1,2))+bulk*logj[:,None,None]*fit
 grad=np.empty((len(cells),4,3));grad[:,1:]=inv;grad[:,0]=-inv.sum(axis=1)
 force=-volume[:,None,None]*(p@np.swapaxes(grad,1,2)).swapaxes(1,2)
 density=shear*(tr-logj+.5*(h*h).sum(axis=(1,2)))+.5*bulk*logj*logj
 return force,density*volume,volume

def scatter(cells,values,n):
 out=np.zeros((n,)+values.shape[2:])
 np.add.at(out,cells.reshape(-1),values.reshape((-1,)+values.shape[2:]))
 return out

def calibrated_force_error(calibration):
 """Reject an unexcited/nonfinite calibration rather than dividing by a floor."""
 native=float(calibration["native_free_force_l1_N"])
 error=float(calibration["analytical_minus_native_free_force_l1_N"])
 if not np.isfinite(native) or native<=0 or not np.isfinite(error) or error<0:
  raise ValueError("force calibration requires a positive finite native signal")
 return error/native

class RegionalResponseBudget:
 """Conservative fixed-field bounds; never a native fidelity certificate.

 Triangle inequalities bound both resultant and nodal-L1 force changes. Sum
 absolute patch energy changes so cancellations cannot conceal local changes.
 Force ownership is the captured node region, independent of cell donor labels.
 """
 def __init__(self,force_reference,energy_reference):
  self.force={int(k):float(v) for k,v in force_reference.items()}
  self.energy={int(k):float(v) for k,v in energy_reference.items()}
  if any(not np.isfinite(v) or v<0 for v in self.force.values()) or any(not np.isfinite(v) or v<0 for v in self.energy.values()):
   raise ValueError("invalid regional response reference")
  self.force_used={k:0. for k in self.force};self.energy_used={k:0. for k in self.energy}
 def propose(self,force_changes,energy_region,energy_change):
  changes={int(k):float(v) for k,v in force_changes.items()};region=int(energy_region);energy=abs(float(energy_change))
  if region not in self.energy or not np.isfinite(energy) or any(k not in self.force or not np.isfinite(v) or v<0 for k,v in changes.items()):
   raise ValueError("invalid regional response change")
  failures=[]
  for k,v in changes.items():
   bound=self.force_used[k]+v;limit=.01*self.force[k]
   if bound>limit:failures.append(dict(metric="free_force_l1_bound_N",region=k,bound=bound,limit=limit))
  bound=self.energy_used[region]+energy;limit=.05*self.energy[region]
  if bound>limit:failures.append(dict(metric="energy_absolute_change_bound_J",region=region,bound=bound,limit=limit))
  return (changes,region,energy) if not failures else None,failures
 def commit(self,proposal):
  changes,region,energy=proposal
  for k,v in changes.items():self.force_used[k]+=v
  self.energy_used[region]+=energy
 def report(self):
  return dict(scope="Conservative fixed-field preparation only; no native trajectory, reaction, contact, mass or energy admission",force_relative_limit=.01,energy_relative_limit=.05,
   force_owner="captured node anatomical region, tag0 free nodes only",energy_owner="source cell geometric donor",
   force_reference_l1_N=self.force,force_change_l1_bound_N=self.force_used,
   energy_reference_J=self.energy,energy_absolute_change_bound_J=self.energy_used,physical_admission=False)

def regional_force_change(nodes,delta,node_regions,free):
 nodes=np.asarray(nodes);delta=np.asarray(delta,float)
 if delta.shape!=(len(nodes),3) or not np.isfinite(delta).all():raise ValueError("invalid finite nodal force delta")
 out={}
 for region in np.unique(node_regions[nodes]):
  mask=(node_regions[nodes]==region)&free[nodes]
  if np.any(mask):out[int(region)]=float(np.linalg.norm(delta[mask],axis=1).sum())
 return out

def native_geometry_check(checker,checker_sha,volume,proposal,source_sha,proposal_sha,count):
 """Call the existing native certificate; this never advances physical state."""
 if sha(checker)!=checker_sha or sha(proposal)!=proposal_sha:raise ValueError("native geometry checker or payload SHA mismatch")
 argv=[str(checker),"--geometry-proposal",str(volume),str(proposal)]
 start=time.monotonic();result=subprocess.run(argv,capture_output=True,text=True)
 lines=[line.split("=",1)[1] for line in result.stdout.splitlines() if line.startswith("skin_fat_cavity_geometry=")]
 if len(lines)!=1:raise ValueError("native geometry certificate did not return a complete batch: "+result.stderr[-1000:])
 record=json.loads(lines[0]);failures=record.get("failures",[])
 rejected=[row["cavity"] for row in failures]
 if (record.get("source_sha256")!=source_sha or record.get("proposal_sha256")!=proposal_sha or
     record.get("cavities")!=count or record.get("failure_count")!=len(rejected) or
     len(set(rejected))!=len(rejected) or any(type(i) is not int or not 0<=i<count for i in rejected) or
     record.get("native_steps")!=0 or record.get("mass_qualified") is not False or
     result.returncode!=(1 if rejected else 0)):
  raise ValueError("native geometry certificate identity, domain or exit-code mismatch")
 return dict(certificate=record,checker=pin(checker),argv=argv,elapsed_s=time.monotonic()-start),set(rejected)

def native_transport_rejections(trial,payload_sha,count):
 """Replay complete initialization witnesses bound to this exact native trial."""
 plan_path=trial/"install-motion.plan.json";execution_path=trial/"install-motion.execution.json";stderr=trial/"install-motion.stderr.log"
 execution=json.loads(execution_path.read_text());plan=json.loads(plan_path.read_text())
 if (execution["plan_sha256"]!=sha(plan_path) or execution["stderr_sha256"]!=sha(stderr) or
     execution["exit_code"]==0 or execution.get("resource_stop_reason") is not None or
     plan["stages"][0]["environment"].get("NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256")!=payload_sha):
  raise ValueError("native transport witness trial/payload identity mismatch")
 lines=[line.split("=",1)[1] for line in stderr.read_text().splitlines() if line.startswith("skin_fat_cavity_transport_failures=")]
 if len(lines)!=1:raise ValueError("native transport witness is missing or ambiguous")
 r=json.loads(lines[0]);indices=r.get("cavity_indices",[])
 if (r.get("complete") is not True or not indices or len(set(indices))!=len(indices) or
     any(type(i) is not int or not 0<=i<count for i in indices) or
     not np.isfinite(r.get("maximum_transport_distance_m",np.nan)) or r["maximum_transport_distance_m"]<=0):
  raise ValueError("native transport witness is incomplete or outside the cavity domain")
 return dict(certificate=r,plan=pin(plan_path),execution=pin(execution_path),stderr=pin(stderr)),set(indices)

def run(a):
 start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 if sha(a.proposal_report)!=a.proposal_report_sha256:raise ValueError("proposal report mismatch")
 r=json.loads(a.proposal_report.read_text())
 native_input=None;native_rejected=set();transport_input=None;transport_rejected=set()
 if getattr(a,"emit_proposal",False):
  if not getattr(a,"native_geometry_check",None) or not getattr(a,"native_geometry_check_sha256",None):
   raise ValueError("emitting a native proposal requires the pinned existing native geometry checker")
  native_input,native_rejected=native_geometry_check(a.native_geometry_check,a.native_geometry_check_sha256,
   Path(r["input"]["volume"]["path"]),Path(r["output"]["path"]),r["input"]["volume"]["sha256"],r["output"]["sha256"],len(r["patches"]))
  (a.output/"native-input-geometry.json").write_text(json.dumps(native_input,indent=2)+"\n")
  if getattr(a,"native_failed_trial",None):
   transport_input,transport_rejected=native_transport_rejections(a.native_failed_trial,r["output"]["sha256"],len(r["patches"]))
   (a.output/"native-transport-exclusions.json").write_text(json.dumps(transport_input,indent=2)+"\n")
 for v in r["input"].values():
  if sha(v["path"])!=v["sha256"]:raise ValueError("source input mismatch")
 cap=module(a.capture_owner,"retained_native_capture",a.capture_owner_sha256)
 z=cap.capture(Path(r["input"]["reference"]["path"]))
 ob=z["objects"]["BodyParts3D whole-body skin-fat bonded shell"];ni=np.flatnonzero(z["obj"]==ob);ti=np.flatnonzero(z["tobj"]==ob)
 with np.load(r["input"]["volume"]["path"],allow_pickle=False) as src:
  p=src["points_reference_world_m_f64"].astype(np.float32);t=src["volume_tetrahedra"].astype(np.int64)
  mats=src["material_index"];regions=src["geometric_donor_body"]
 if not np.array_equal(z["rest"][ni],p) or not np.array_equal(z["tets"][ti]-ni[0],t) or not np.array_equal(z["treg"][ti],regions):raise ValueError("reference source order mismatch")
 u=z["u"][ni];native_mass=z["mass"][ni];tags=z["tag"][ni];native_force=z["forces"][ni,2]
 params={15:material(a.skin_material,a.skin_material_sha256),16:material(a.fat_material,a.fat_material_sha256)}
 cm=np.loadtxt(r["input"]["mass_csv"]["path"],delimiter=",",skiprows=1,usecols=(1,))
 nodal=np.zeros(len(p));np.add.at(nodal,t.ravel(),np.repeat(cm*.25,4))
 relative_mass=np.abs(nodal-native_mass)/native_mass
 baseline=np.zeros((len(p),3));energy=np.zeros(len(t));vol=np.zeros(len(t))
 for lo in range(0,len(t),8192):
  rows=np.arange(lo,min(lo+8192,len(t)))
  for mat in np.unique(mats[rows]):
   rr=rows[mats[rows]==mat];ff,ee,vv=response(p,t[rr],u,**params[int(mat)])
   np.add.at(baseline,t[rr].ravel(),ff.reshape(-1,3));energy[rr]=ee;vol[rr]=vv
 free=tags==0;force_error=np.linalg.norm(baseline[free]-native_force[free],axis=1)
 calibration=dict(scope="Analytical fixed-field CPU check only; not a native force owner",maximum_mass_relative_difference=float(relative_mass.max()),
  native_free_force_l1_N=float(np.linalg.norm(native_force[free],axis=1).sum()),analytical_free_force_l1_N=float(np.linalg.norm(baseline[free],axis=1).sum()),
  analytical_minus_native_free_force_l1_N=float(force_error.sum()),analytical_minus_native_free_force_max_N=float(force_error.max()))
 base={}
 for region in np.unique(regions):
  rr=regions==region;nodes=np.flatnonzero(z["region"][ni]==region)
  base[int(region)]=dict(energy_J=float(energy[rr].sum()),native_free_force_l1_N=float(np.linalg.norm(native_force[nodes],axis=1).sum()))
 patches=[]
 for index,patch in enumerate(r["patches"]):
  rr=np.array(patch["old_rows"]);old=t[rr];new=np.array(patch["new_cells"]);nodes=np.unique(old)
  oldf,olde,oldv=response(p,old,u,**params[int(mats[rr[0]])]);newf,newe,newv=response(p,new,u,**params[int(mats[rr[0]])])
  oldlocal=np.searchsorted(nodes,old);newlocal=np.searchsorted(nodes,new)
  of=scatter(oldlocal,oldf,len(nodes));nf=scatter(newlocal,newf,len(nodes))
  total=float(cm[rr].sum());nm=total*newv/newv.sum()
  om=scatter(oldlocal,np.repeat((cm[rr]*.25)[:,None,None],4,axis=1),len(nodes)).ravel()
  nmm=scatter(newlocal,np.repeat((nm*.25)[:,None,None],4,axis=1),len(nodes)).ravel()
  mass_delta=nmm-om;fd=nf-of;free_local=tags[nodes]==0
  if not np.all(mats[rr]==mats[rr[0]]) or not np.all(regions[rr]==regions[rr[0]]):raise ValueError("cavity crosses material or donor ownership")
  rows=dict(index=index,material=int(mats[rr[0]]),donor=int(regions[rr[0]]),old_rows=rr.tolist(),
   regional_free_force_change_l1_bound_N=regional_force_change(nodes,fd,z["region"][ni],free),
   old_energy_J=float(olde.sum()),new_energy_J=float(newe.sum()),
   relative_patch_energy_change=float(abs(newe.sum()-olde.sum())/max(abs(olde.sum()),1e-300)),
   maximum_nodal_mass_relative_change=float(np.max(np.abs(mass_delta)/native_mass[nodes])),
   nodal_mass_l1_change_kg=float(np.abs(mass_delta).sum()),
   fixed_field_force_delta_l1_N=float(np.linalg.norm(fd,axis=1).sum()),
   fixed_field_free_force_delta_l1_N=float(np.linalg.norm(fd[free_local],axis=1).sum()),
   fixed_field_free_force_resultant_delta_N=np.sum(fd[free_local],axis=0).tolist(),
   force_change_over_patch_force_l1=float(np.linalg.norm(fd,axis=1).sum()/max(np.linalg.norm(of,axis=1).sum(),1e-300)))
  patches.append(rows)
 out=dict(status="MEASURED_FIXED_FIELD_CAVITY_RESPONSE_NOT_PHYSICAL_ADMISSION",input_report=pin(a.proposal_report),
  owners={str(x):sha(x) for x in [Path(__file__),a.capture_owner,a.skin_material,a.fat_material]},
  parameter_policy="Unchanged pinned native material defaults, MPa converted to Pa; no calibration or fitted parameter",
  baseline_calibration=calibration,regional_baseline=base,patches=patches,
  reference_field_not_independent_validation=True,physical_admission=False,native_steps=0,elapsed_s=time.monotonic()-start)
 (a.output/"report.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
 print(json.dumps(dict(status=out["status"],baseline_calibration=calibration,cavities=len(patches),elapsed_s=out["elapsed_s"]),indent=2),flush=True)
 if getattr(a,"emit_proposal",False):
  error=calibrated_force_error(calibration)
  if error>.01:raise ValueError("native force calibration failed; no proposal emitted")
  builder_path=Path(__file__).with_name("build_human_conditioned_cavities.py")
  mesh_path=Path(__file__).with_name("simplify_human_skin_fat_boundary.py")
  builder=module(builder_path,"existing_cavity_payload",r["owners"][str(builder_path)])
  mesh=module(mesh_path,"existing_deformation_screen",r["owners"][str(mesh_path)])
  with np.load(r["input"]["volume"]["path"],allow_pickle=False) as source:
   labels=np.column_stack([source[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]])
  ref=r["input"]["reference"]
  field,receipt=mesh.load_reference_deformation(Path(ref["path"]),ref["sha256"],p,t,labels)
  gate=mesh.ReferenceDeformationGate(p,t,field,labels[:,1],receipt)
  force_reference=regional_force_change(np.arange(len(p)),native_force,z["region"][ni],free)
  budget=RegionalResponseBudget(force_reference,{k:v["energy_J"] for k,v in base.items()})
  kept=[];excluded=[];indices=[];response_rejections=[]
  for index,patch in enumerate(r["patches"]):
   if index in transport_rejected:
    excluded.append(index);response_rejections.append(dict(index=index,failures=[dict(metric="native_transport_radius",reason="complete source-flow witness from exact failed native input; frozen radius unchanged")]));continue
   if index in native_rejected:
    excluded.append(index);response_rejections.append(dict(index=index,failures=[dict(metric="native_geometry",reason=next(row["reason"] for row in native_input["certificate"]["failures"] if row["cavity"]==index))]));continue
   measured=patches[index]
   response_proposal,failures=budget.propose(measured["regional_free_force_change_l1_bound_N"],measured["donor"],measured["new_energy_J"]-measured["old_energy_J"])
   if response_proposal is None:
    excluded.append(index);response_rejections.append(dict(index=index,failures=failures));continue
   rr=np.array(patch["old_rows"]);new=np.array(patch["new_cells"])
   padded=t[rr].copy();padded[:len(new)]=new;live=np.arange(len(rr))<len(new)
   accepted=gate.propose(rr,padded,live)
   if accepted is None:excluded.append(index);continue
   gate.commit(rr,live,accepted);budget.commit(response_proposal);kept.append(patch);indices.append(index)
  if not kept:raise ValueError("no cavities pass the unchanged reference gate")
  payload=a.output/"fixed-boundary-volume-proposal.npz"
  np.savez_compressed(payload,**builder.cavity_payload(r["input"]["volume"]["sha256"],len(t),kept))
  native_output,rejected=native_geometry_check(a.native_geometry_check,a.native_geometry_check_sha256,
   Path(r["input"]["volume"]["path"]),payload,r["input"]["volume"]["sha256"],sha(payload),len(kept))
  (a.output/"native-output-geometry.json").write_text(json.dumps(native_output,indent=2)+"\n")
  if rejected:raise ValueError("screened proposal failed the existing native geometry certificate")
  r["native_geometry"]=dict(input=native_input,output=native_output)
  if transport_input is not None:r["native_transport_exclusions"]=transport_input
  history={k:r.pop(k) for k in ["eligible_seed_cells","tested_edges","rejections","elapsed_s","replay","reference_rebinding_diagnostic"] if k in r}
  r["source_report_history"]=history
  r["owners"][str(Path(__file__))]=sha(Path(__file__))
  r.update(patches=kept,passing_cavities=len(kept),output=pin(payload),reference_deformation=gate.report(),elapsed_s=time.monotonic()-start,
   reference_screen=dict(source_report=pin(a.proposal_report),response_report=pin(a.output/"report.json"),
    excluded_cavity_indices=excluded,retained_input_cavity_indices=indices,calibration_relative_force_l1_error=error,
    owner=pin(Path(__file__)),unchanged_kinematic_strain_relative_gate=.05,
    response_budget=budget.report(),response_rejections=response_rejections),
   status="PASS_FIXED_BOUNDARY_REFERENCE_SCREENED_SUBSET_PROPOSAL",physical_admission=False,native_steps=0)
  (a.output/"proposal-report.json").write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
  print(json.dumps(dict(status=r["status"],passing_cavities=len(kept),excluded=excluded,output=r["output"])),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ["proposal-report","capture-owner","skin-material","fat-material"]:
  ap.add_argument("--"+name,type=Path,required=True);ap.add_argument("--"+name+"-sha256",required=True)
 ap.add_argument("--native-failed-trial",type=Path,help="Exclude complete frozen-radius failure witnesses bound to this exact input payload; all remaining admission repeats")
 ap.add_argument("--native-geometry-check",type=Path);ap.add_argument("--native-geometry-check-sha256")
 ap.add_argument("--emit-proposal",action="store_true",help="Replay only cavities passing 1% fixed-field force bounds,5% energy bounds and the existing5% strain gate after force calibration; physical admission remains mandatory")
 ap.add_argument("--output",type=Path,required=True);run(ap.parse_args())
