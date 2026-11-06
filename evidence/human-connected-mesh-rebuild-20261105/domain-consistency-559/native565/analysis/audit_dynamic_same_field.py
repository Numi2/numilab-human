"""Postprocess existing endpoint velocity/mass/residual channels. No simulation."""
from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=Path(__file__).resolve().parent
sys.path.insert(0,str(W))
import compare_remeshed_native_310 as parser
from compare_matched_deformation_339 import retained_node_map
source=Path(parser.__file__);text=source.read_text();assert hashlib.sha256(source.read_bytes()).hexdigest()=='0181dd99e2dc5c9c47c0e9c725d6762edee5dd41ca2bd4329f328e42a85d2902'
start=text.index('def capture(');end=text.index('def strain_stats(',start);func=text[start:end]
old='forces=np.zeros((n,len(CHANNELS),3));tag=np.zeros(n)';assert func.count(old)==1
func=func.replace(old,old+';velocity=np.zeros_like(rest);residual=np.zeros_like(rest)')
old='    for c,prefix in enumerate(CHANNELS):forces[i,c]=[float(row[prefix+"_"+a+"_N"]) for a in "xyz"]';assert func.count(old)==1
func=func.replace(old,old+'\n    velocity[i]=[float(row["v"+a]) for a in "xyz"]\n    residual[i]=[float(row["residual_impulse_"+a+"_Ns"]) for a in "xyz"]')
old='tag=tag,tets=tets';assert func.count(old)==1;func=func.replace(old,'tag=tag,velocity=velocity,residual=residual,tets=tets')
exec(compile(func,str(source)+'::read_existing_dynamic_fields','exec'),vars(parser))
paths=[O.parent/'control/install-motion.stdout.log.gz',O.parent/'candidate/install-motion.stdout.log.gz']
a,b=map(parser.capture,paths);mapping,map_receipt=retained_node_map(a,b)
assert np.isfinite(a['velocity']).all() and np.isfinite(b['velocity']).all() and np.isfinite(a['residual']).all() and np.isfinite(b['residual']).all()
assert a['header']['runtime_dt_s']==b['header']['runtime_dt_s'];dt=float(a['header']['runtime_dt_s']);assert dt>0
names={v:k for k,v in b['objects'].items()}
def witness(i):
 j=int(mapping[i]);return dict(candidate_node=int(i),reference_node=j,object=names[int(b['obj'][i])],region=int(b['region'][i]),tag=float(b['tag'][i]),rest_m=b['rest'][i].tolist(),displacement_difference_m=float(np.linalg.norm(b['u'][i]-a['u'][j])),velocity_difference_m_s=float(np.linalg.norm(b['velocity'][i]-a['velocity'][j])),reference_mass_kg=float(a['mass'][j]),candidate_mass_kg=float(b['mass'][i]),reference_retained_residual_Ns=a['residual'][j].tolist(),candidate_retained_residual_Ns=b['residual'][i].tolist())
du=np.linalg.norm(b['u']-a['u'][mapping],axis=1);dv=np.linalg.norm(b['velocity']-a['velocity'][mapping],axis=1);dm=b['mass']-a['mass'][mapping]
relative=np.abs(dm)/a['mass'][mapping];removed=np.ones(len(a['mass']),bool);removed[mapping]=False
regions={}
for region in np.unique(b['region'][b['region']>=0]):
 mask=(b['region']==region)&(b['tag']==0);prior=(a['region']==region)&(a['tag']==0)
 regions[str(int(region))]={}
 for name,z,m in [('reference',a,prior),('candidate',b,mask)]:
  values=z['residual'][m];force=z['forces'][m,2];mass=z['mass'][m];vel=z['velocity'][m]
  regions[str(int(region))][name]=dict(free_nodes=int(m.sum()),free_mass_kg=float(mass.sum()),retained_residual_impulse_l2_Ns=float(np.linalg.norm(values)),retained_residual_impulse_l1_Ns=float(np.linalg.norm(values,axis=1).sum()),retained_residual_equivalent_force_l1_N=float(np.linalg.norm(values,axis=1).sum()/dt),free_internal_force_l1_N=float(np.linalg.norm(force,axis=1).sum()),free_FEM_kinetic_energy_J=float((.5*mass*np.einsum('ij,ij->i',vel,vel)).sum()))
# Exploratory fixed-field inertia probe; no acceptance thresholds changed.
# Evaluate both lumped mass fields at the reference endpoint velocity so the
# measured difference excludes candidate trajectory drift.
inertia={}
for region in [-1]+[int(x) for x in np.unique(b['region'][b['region']>=0])]:
 ca=np.ones(len(b['mass']),bool) if region<0 else b['region']==region
 ra=np.ones(len(a['mass']),bool) if region<0 else a['region']==region
 old_m=a['mass'][ra];old_v=a['velocity'][ra]
 new_m=b['mass'][ca];new_v=a['velocity'][mapping[ca]]
 old_k=float((.5*old_m*np.einsum('ij,ij->i',old_v,old_v)).sum())
 new_k=float((.5*new_m*np.einsum('ij,ij->i',new_v,new_v)).sum())
 old_p=(old_m[:,None]*old_v).sum(axis=0);new_p=(new_m[:,None]*new_v).sum(axis=0)
 scale=float((old_m*np.linalg.norm(old_v,axis=1)).sum())
 inertia[str(region)]=dict(reference_kinetic_energy_J=old_k,candidate_mass_on_reference_velocity_energy_J=new_k,relative_kinetic_energy_change=abs(new_k-old_k)/old_k if old_k else None,reference_mass_kg=float(old_m.sum()),candidate_mass_kg=float(new_m.sum()),momentum_action_difference_kg_m_s=(new_p-old_p).tolist(),relative_momentum_action_difference_over_reference_nodal_l1=float(np.linalg.norm(new_p-old_p))/scale if scale else None)
r=dict(scope='Existing endpoint observations only. Retained mechanical residual buffer is not independently reassembled here; no residual-based physical admission. Retained-node mass differences are not a conservative full-field comparison.',runtime_dt_s=dt,input_pins=[dict(path=str(p),sha256=parser.sha(p)) for p in paths],parser_owner=dict(path=str(source),sha256=parser.sha(source)),mapping=map_receipt,maximum_displacement_witness=witness(int(du.argmax())),maximum_velocity_witness=witness(int(dv.argmax())),maximum_relative_retained_mass_change_witness=witness(int(relative.argmax())),retained_mass_difference_l1_kg=float(np.abs(dm).sum()),removed_nodes=int(removed.sum()),removed_reference_mass_kg=float(a['mass'][removed].sum()),regions=regions,fixed_reference_velocity_inertia_probe=inertia,inertia_probe_admission=False,physical_admission=False)
(O/'dynamic-channel-diagnostic.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['regions','mapping']},indent=2),flush=True)

# Separate the changed approximation from the changed native trajectory by
# transferring the exact same reference displacement to the retained nodes.
reference_strain=parser.strain_stats(a)
transferred=dict(b);transferred['u']=a['u'][mapping]
transferred_strain=parser.strain_stats(transferred)
failed=[];worst=0.0
for region in reference_strain:
 for key in ['volume_weighted_strain_norm','volume_weighted_strain_squared','maximum_green_strain_norm']:
  x=reference_strain[region][key];y=transferred_strain[region][key]
  change=abs(y-x)/abs(x) if x else (0.0 if y==0 else None)
  if change is not None:worst=max(worst,change)
  if change is None or change>.05:failed.append(dict(region=region,metric=key,reference=x,transferred=y,relative_change=change))
probe=dict(scope='Exploratory same-reference-displacement counterfactual at four roots; not actual native trajectory or material energy admission.',failed_regional_strain_comparisons=failed,maximum_relative_change=worst,reference_regional_strain=reference_strain,transferred_regional_strain=transferred_strain,physical_admission=False)
(O/'same-field-strain-counterfactual.json').write_text(json.dumps(probe,indent=2,allow_nan=False)+'\n')
print('same_field_strain',len(failed),worst,flush=True)
