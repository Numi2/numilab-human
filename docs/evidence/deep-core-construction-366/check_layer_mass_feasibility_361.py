#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'
FILES={'O':B/'minimum-thickness-minkowski-328/attempt-006-anchored-local-topology/run-018-compose-347-onto-conditioned/candidate-O-closed-f64-geometry.npz','I':B/'minimum-thickness-minkowski-328/attempt-004-prebuffer-simplify/candidate-I-after-fat-f64.npz','D':B/'dnew-vs-i296-inventory-001/typed-D-62000100/run-002/D-closed-csg-only.npz'}
MASS=P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
OUT=P/'layer-mass-feasibility-361'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False);rows={}
 for role,p in FILES.items():
  with np.load(p,allow_pickle=False) as z:
   x=np.asarray(z['points_world_m_f64'] if 'points_world_m_f64' in z.files else z['points_world_m'],np.float64)
   f=np.asarray(z['faces_f64'] if 'faces_f64' in z.files else z['faces'],np.int64)
  # Local origin reduces cancellation; closed mesh translation invariance is required.
  edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);keys=np.sort(edges,axis=1)
  _,counts=np.unique(keys,axis=0,return_counts=True)
  if np.any(counts!=2):raise ValueError(f'{role} not closed edge manifold')
  q=x-x.mean(0);a=q[f];terms=np.einsum('fa,fa->f',a[:,0],np.cross(a[:,1],a[:,2]))/6
  vol=float(np.sum(terms,dtype=np.longdouble))
  rows[role]={'path':str(p),'sha256':sha(p),'vertices':len(x),'faces':len(f),'signed_volume_m3':vol}
 m=np.genfromtxt(MASS,delimiter=',',names=True);masses={int(k):float(m['mass_kg'][m['material_index']==k].sum()) for k in np.unique(m['material_index'])}
 checks=[]
 for name,mat,outer,inner,bounds in [('skin',15,'O','I',[900.,1100.]),('fat',16,'I','D',[850.,950.])]:
  v=rows[outer]['signed_volume_m3']-rows[inner]['signed_volume_m3'];mass=masses[mat];rho=mass/v
  checks.append({'layer':name,'material_index':mat,'candidate_shell_volume_m3':v,'retained_mass_kg':mass,'implied_mean_density_kg_m3':rho,'material_density_bounds_kg_m3':bounds,'volume_required_by_density_bounds_m3':[mass/bounds[1],mass/bounds[0]],'global_necessary_density_feasibility_pass':bool(bounds[0]<=rho<=bounds[1])})
 r={'status':'NECESSARY_GEOMETRIC_MASS_FEASIBILITY_DIAGNOSTIC','boundaries':rows,'parent_mass_packet':str(MASS),'parent_mass_packet_sha256':sha(MASS),'mass_by_material_kg':masses,'layers':checks,'limitations':['Signed-volume differences assume correctly nested oriented solids; this diagnostic does not certify nesting or mesh validity.','Global density feasibility is necessary but not sufficient for local donor/region spatial mass transfer.','No geometry or physical state changed; no steps run.'],'physical_admission':False,'elapsed_s':time.monotonic()-t}
 (OUT/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
