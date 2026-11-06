from pathlib import Path
import importlib.util,json,hashlib
O=Path(__file__).resolve().parent
plan=json.loads((O/'plan.json').read_text())
for path,digest in plan['input_pins'].items():
 with open(path,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,path
spec=importlib.util.spec_from_file_location('current_skin_fat_compiler',O/'compiler-source.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
owner.S={role:Path(path) for role,path in plan['typed_surfaces'].items()}
import numpy as np
cap_spec=importlib.util.spec_from_file_location('reference_capture',plan['capture_owner'])
cap=importlib.util.module_from_spec(cap_spec);cap_spec.loader.exec_module(cap)
response_spec=importlib.util.spec_from_file_location('reference_material_response',plan['response_owner'])
response=importlib.util.module_from_spec(response_spec);response_spec.loader.exec_module(response)
BaseGate=owner.ReferenceDeformationGate
class ResponseGate(BaseGate):
 def __init__(self,p,t,u,regions,receipt):
  super().__init__(p,t,u,regions,receipt)
  self.t=t;self.response_rejected=0;self.response_witnesses=[]
  z=cap.capture(Path(receipt['path']));ob=z['objects']['BodyParts3D whole-body skin-fat bonded shell'];ni=np.flatnonzero(z['obj']==ob);ti=np.flatnonzero(z['tobj']==ob)
  with np.load(plan['resume_volume']) as source:
   self.materials=source['material_index'];labels=np.column_stack([self.materials,source['geometric_donor_body'],source['anatomical_region_id']])
  exact_u,mapping=owner.match_reference_deformation(p,t,labels,z['rest'][ni],ni,z['tets'][ti],z['treg'][ti],z['u'][ni],require_same_donor=False)
  assert np.array_equal(exact_u,u)
  active=np.unique(t);self.node_regions=np.full(len(p),-1);self.node_regions[active]=z['region'][ni]
  self.free=np.zeros(len(p),bool);self.free[active]=z['tag'][ni]==0
  native=np.zeros((len(p),3));native[active]=z['forces'][ni,2]
  self.params={int(k):response.material(Path(v['path']),v['sha256']) for k,v in plan['materials'].items()}
  baseline=np.zeros_like(native);energy={}
  for start in range(0,len(t),8192):
   rows=np.arange(start,min(start+8192,len(t)))
   for mat in np.unique(self.materials[rows]):
    rr=rows[self.materials[rows]==mat];ff,ee,vv=response.response(p,t[rr],u,**self.params[int(mat)])
    np.add.at(baseline,t[rr].ravel(),ff.reshape(-1,3))
    for r in np.unique(regions[rr]):
     key=int(r);energy[key]=energy.get(key,0.)+float(ee[regions[rr]==r].sum())
  self.calibration=dict(native_free_force_l1_N=float(np.linalg.norm(native[self.free],axis=1).sum()),analytical_minus_native_free_force_l1_N=float(np.linalg.norm((baseline-native)[self.free],axis=1).sum()))
  self.calibration_relative=response.calibrated_force_error(self.calibration)
  assert self.calibration_relative<=.01,'native force calibration failed'
  force=response.regional_force_change(np.arange(len(p)),native,self.node_regions,self.free)
  analytical=response.regional_force_change(np.arange(len(p)),baseline,self.node_regions,self.free)
  # Tighter construction budget: both native and analytical reference norms,
  # with1percent of the1percent budget reserved for arithmetic/calibration.
  force={k:.99*min(v,analytical[k]) for k,v in force.items()}
  self.budget=response.RegionalResponseBudget(force,energy)
  print(json.dumps(dict(response_calibration_relative=self.calibration_relative,reference_force_regions=len(force))),flush=True)
 def propose(self,rows,new,live):
  strain=super().propose(rows,new,live)
  if strain is None:return None
  nodes=np.unique(self.t[rows]);delta=np.zeros((len(nodes),3));energy_delta={}
  for mat in np.unique(self.materials[rows]):
   mask=self.materials[rows]==mat;rr=rows[mask];newmask=mask&live
   of,oe,ov=response.response(self.p,self.t[rr],self.u,**self.params[int(mat)])
   nf,ne,nv=response.response(self.p,new[newmask],self.u,**self.params[int(mat)])
   np.add.at(delta,np.searchsorted(nodes,self.t[rr]).ravel(),-of.reshape(-1,3))
   np.add.at(delta,np.searchsorted(nodes,new[newmask]).ravel(),nf.reshape(-1,3))
   for r in np.unique(self.regions[rr]):
    key=int(r);change=float(ne[self.regions[rows[newmask]]==r].sum()-oe[self.regions[rr]==r].sum())
    energy_delta[key]=energy_delta.get(key,0.)+abs(change)
  changes=response.regional_force_change(nodes,delta,self.node_regions,self.free)
  proposals=[];failures=[]
  for j,(r,e) in enumerate(sorted(energy_delta.items())):
   value,bad=self.budget.propose(changes if j==0 else {},r,e)
   proposals.append(value);failures.extend(bad)
  if failures:
   self.response_rejected+=1
   if len(self.response_witnesses)<32:self.response_witnesses.append(dict(source_cell_rows=rows.tolist(),failures=failures))
   return None
  return strain,proposals
 def commit(self,rows,live,proposal):
  strain,proposals=proposal;super().commit(rows,live,strain)
  for value in proposals:self.budget.commit(value)
 def report(self):
  result=super().report();result['fixed_field_response']=dict(calibration=self.calibration,calibration_relative=self.calibration_relative,budget=self.budget.report(),force_budget_reference_policy='0.99 times minimum native and analytical regional force; physical gate remains1percent',rejected_operations=self.response_rejected,failure_witnesses=self.response_witnesses)
  return result
owner.ReferenceDeformationGate=ResponseGate
owner.main()
