from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');H=Path('/Users/numi/Work/human-checkpoint-owner-20261105');O=Path(__file__).resolve().parent
start=time.monotonic()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
cap=module(W/'matched-endpoint-pair-509/compare_endpoint_channels.py','capture')
mesh=module(H/'tools/simplify_human_skin_fat_boundary.py','mesh')
response=module(H/'tools/screen_human_cavity_response.py','response')
plan=json.loads((W/'star-cavity-response-424/analysis-plan.json').read_text());args=plan['stages'][0]['argv']
params={}
for mat,role in [(15,'skin'),(16,'fat')]:
 path=Path(args[args.index('--'+role+'-material')+1]);digest=args[args.index('--'+role+'-material-sha256')+1]
 params[mat]=response.material(path,digest)
source=W/'captured-fat-sliver-495/proposal/candidate-volume.npz';proposal=W/'early-state-screen-568/proposal/candidate-volume.npz';capture=Path(json.loads((W/'early-state-screen-568/plan.json').read_text())['matched_reference_fields'][int(__import__('sys').argv[1])]['path'])
z=cap.capture(capture);ob=z['objects']['BodyParts3D whole-body skin-fat bonded shell'];ni=np.flatnonzero(z['obj']==ob);ti=np.flatnonzero(z['tobj']==ob)
with np.load(source) as a:p=a['points_reference_world_m_f64'].astype(np.float32);t=a['volume_tetrahedra'];m=a['material_index'];donor=a['geometric_donor_body'];labels=np.column_stack([m,donor,a['anatomical_region_id']])
u,receipt=mesh.match_reference_deformation(p,t,labels,z['rest'][ni],ni,z['tets'][ti],z['treg'][ti],z['u'][ni],require_same_donor=False)
active=np.unique(t);node_regions=np.full(len(p),-1);node_regions[active]=z['region'][ni];free=np.zeros(len(p),bool);free[active]=z['tag'][ni]==0
native=np.zeros((len(p),3));native[active]=z['forces'][ni,2]
def evaluate(t,m,donor):
 force=np.zeros((len(p),3));energy={};volume={}
 for start in range(0,len(t),8192):
  rows=np.arange(start,min(start+8192,len(t)))
  for mat in np.unique(m[rows]):
   rr=rows[m[rows]==mat];ff,ee,vv=response.response(p,t[rr],u,**params[int(mat)])
   np.add.at(force,t[rr].ravel(),ff.reshape(-1,3))
   for r in np.unique(donor[rr]):
    mask=donor[rr]==r;key=int(r);energy[key]=energy.get(key,0.)+float(ee[mask].sum());volume[key]=volume.get(key,0.)+float(vv[mask].sum())
 return force,energy,volume
base,be,bv=evaluate(t,m,donor)
with np.load(proposal) as a:
 assert np.array_equal(a['points_reference_world_m_f64'],p.astype(float))
 trial,te,tv=evaluate(a['volume_tetrahedra'],a['material_index'],a['geometric_donor_body'])
calibration=dict(native_free_force_l1_N=float(np.linalg.norm(native[free],axis=1).sum()),analytical_minus_native_free_force_l1_N=float(np.linalg.norm((base-native)[free],axis=1).sum()))
calerror=response.calibrated_force_error(calibration)
regions=[]
for r in np.unique(node_regions[free]):
 mask=(node_regions==r)&free;baseline=float(np.linalg.norm(base[mask],axis=1).sum());change=float(np.linalg.norm((trial-base)[mask],axis=1).sum())
 regions.append(dict(region=int(r),baseline_free_force_l1_N=baseline,force_delta_l1_N=change,relative_force_delta=change/baseline if baseline else (0. if change==0 else None),passes=change<=.01*baseline))
energy=[dict(region=r,baseline_J=be[r],candidate_J=te.get(r,0.),passes=abs(te.get(r,0.)-be[r])<=.05*abs(be[r])) for r in be]
out=dict(scope='Calibrated fixed matched reference field only; no trajectory or native admission',input_pins={str(q):sha(q) for q in [source,proposal,capture,H/'tools/screen_human_cavity_response.py',H/'tools/simplify_human_skin_fat_boundary.py']},source_mapping=receipt,calibration=calibration,calibration_relative=calerror,calibration_pass=calerror<=.01,force=regions,energy=energy,force_failures=sum(not r['passes'] for r in regions),energy_failures=sum(not r['passes'] for r in energy),elapsed_s=time.monotonic()-start,physical_admission=False)
(O/('independent-field-'+__import__('sys').argv[1]+'.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['calibration_relative','calibration_pass','force_failures','energy_failures','elapsed_s']}),flush=True)
