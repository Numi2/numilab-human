from pathlib import Path
import hashlib, json, time
from fractions import Fraction
import numpy as np

BASE = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT = BASE/'global-mass-volume-preflight-532/run001'
FILES = {
 'Skin': BASE/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz',
 'Fat': BASE/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/volume-transfer-500um-001/compose-comp0-54138-001/fat-D500-comp0-kernel-fan-composed-volume-proposal.npz',
}
MASS = {'Skin':3.4525704454739734,'Fat':10.487108517986506}
BOUNDS = {'Skin':(900.0,1100.0),'Fat':(850.0,950.0)}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def ipoint(row):
 out=[]
 for v in row:
  n,d=float(v).as_integer_ratio()
  e=d.bit_length()-1
  if d != (1<<e) or e>149: raise ValueError('not a finite F32 dyadic on 2^-149 lattice')
  out.append(n << (149-e))
 return tuple(out)

def det4(a,b,c,d):
 u=tuple(b[i]-a[i] for i in range(3)); v=tuple(c[i]-a[i] for i in range(3)); w=tuple(d[i]-a[i] for i in range(3))
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])

start=time.monotonic(); results={}
for role,path in FILES.items():
 with np.load(path,allow_pickle=False) as z:
  points=z['points_world_m']; tets=z['tetrahedra'].astype(np.int64,copy=False)
 if points.dtype != np.float32: raise ValueError(f'{role} points are not F32')
 pints=[ipoint(row) for row in points]
 total=0; min_det=None; max_det=None; neg=zero=0
 for tet in tets:
  d=det4(pints[int(tet[0])],pints[int(tet[1])],pints[int(tet[2])],pints[int(tet[3])])
  neg += d<0; zero += d==0
  if min_det is None or d<min_det:min_det=d
  if max_det is None or d>max_det:max_det=d
  total += d
 vol=Fraction(total,6*(1<<447))
 v=float(vol); m=MASS[role]; lo,hi=BOUNDS[role]
 results[role]={'input_path':str(path),'input_sha256':sha(path),'point_count':int(len(points)),'tet_count':int(len(tets)),'exact_dyadic_positive_tets':int(len(tets)-neg-zero),'exact_dyadic_negative_tets':int(neg),'exact_dyadic_zero_tets':int(zero),'exact_signed_det6_sum_lattice':str(total),'exact_signed_volume_m3_fraction':f'{vol.numerator}/{vol.denominator}','exact_signed_volume_m3':v,'minimum_exact_signed_det6_lattice':str(min_det),'maximum_exact_signed_det6_lattice':str(max_det),'source_mass_kg':m,'density_bounds_kg_m3':[lo,hi],'implied_uniform_density_kg_m3':m/v,'allowed_mass_kg':[lo*v,hi*v],'required_volume_m3_for_mass':[m/hi,m/lo],'aggregate_role_feasible':bool(lo*v <= m <= hi*v),'mass_excess_over_max_kg':max(0.0,m-hi*v),'elapsed_s':time.monotonic()-start}
volsum=sum(r['exact_signed_volume_m3'] for r in results.values()); masssum=sum(MASS.values())
results['combined']={'exact_signed_volume_m3':volsum,'source_mass_kg':masssum,'implied_uniform_density_kg_m3':masssum/volsum,'elapsed_s':time.monotonic()-start}
report={'schema':'numi.human.current-role-global-density-volume-preflight.v1','status':'SOURCE_ONLY_EXACT_DYADIC_VOLUME_DENSITY_PREFLIGHT','physical_admission':False,'native_steps':0,'mass_allocation':'NOT_RUN','density_bounds_are_per_role':True,'method':'exact F32 dyadic determinant on the 2^-149 integer lattice; signed tetrahedron determinant sum divided by 6*2^447','input_geometry_scope':'Skin531 and Fat54138/D500; Fat I528-composed successor not yet included','results':results,'limitations':['Global role totals only; no regional or receiver-level feasibility proof.','Fat target is pre-I528 composition and must be refreshed against the final paired geometry pin.','No mass allocation, physics run, or admission claim.']}
out=OUT/'global-density-volume-report.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report_path':str(out),'report_sha256':sha(out),'results':results},sort_keys=True))
