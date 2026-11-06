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
paths=[W/'targeted-fat-native-499/install-motion.stdout.log.gz',W/'joint-response-native-546/install-motion.stdout.log.gz']
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
r=dict(scope='Existing endpoint observations only. Retained mechanical residual buffer is not independently reassembled here; no residual-based physical admission. Retained-node mass differences are not a conservative full-field comparison.',runtime_dt_s=dt,input_pins=[dict(path=str(p),sha256=parser.sha(p)) for p in paths],parser_owner=dict(path=str(source),sha256=parser.sha(source)),mapping=map_receipt,maximum_displacement_witness=witness(int(du.argmax())),maximum_velocity_witness=witness(int(dv.argmax())),maximum_relative_retained_mass_change_witness=witness(int(relative.argmax())),retained_mass_difference_l1_kg=float(np.abs(dm).sum()),removed_nodes=int(removed.sum()),removed_reference_mass_kg=float(a['mass'][removed].sum()),regions=regions,physical_admission=False)
(O/'dynamic-channel-diagnostic.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['regions','mapping']},indent=2),flush=True)
