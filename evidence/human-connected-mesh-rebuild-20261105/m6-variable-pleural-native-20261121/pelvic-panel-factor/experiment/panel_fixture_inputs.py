import csv,json,sys
from pathlib import Path
import numpy as np
from audit_complete_pelvic_forces import material
a,p,z,out=map(Path,sys.argv[1:])
d=json.loads((a/'manifest.json').read_text());dims=d['regional_factor']
np.array(json.loads((p/'manifest.json').read_text())['offsets'],'u4').tofile(p/'offsets.bin')
s='#include <metal_stdlib>\nusing namespace metal;\nconstant uint pelvicFactorCapacity='+str(max(x['coordinates'] for x in dims))+'u;\n'
s+='constant uint4 pelvicFactorDims[2]={'+','.join('uint4(%d,%d,%d,%d)'%(x['coordinate_start'],x['coordinates'],*x['lower']) for x in dims)+'};\n'
(p/'bench.metal').write_text(s+Path('pelvic_panel_factor.metalinc').read_text())
with (z/'accepted-pelvic-state.csv').open() as f:rows=list(csv.DictReader(f))
def field(k):return {int(r['index']):np.array([r[c] for c in 'xyzw'],'f4').astype(float) for r in rows if r['field']==k}
delta=field('material_delta');n=len(delta);x=np.stack([delta[k][:3] for k in range(n)])
cells=np.fromfile(a/'material-cells.bin','u4').reshape(-1,4);inv=np.fromfile(a/'pelvic-reference-inverse.bin','f4').reshape(-1,3,4)[:,:,:3].astype(float);vol=np.fromfile(a/'pelvic-cell-volume.bin','f4').astype(float)
force=material(x,cells,inv,vol,12000.,120000.,True)[0]
ports=np.fromfile(a/'pool-material-points.bin','f4').reshape(-1,7,4);par=field('reverse_parent_nodes');weights=field('reverse_shape')
for k,v in field('normal_and_force').items():
    if v[3]:
        nodes=par[k].astype(int) if k in par else ports[k,5].view('u4')
        np.add.at(force,nodes,-weights.get(k,ports[k,6])[:,None]*v[:3]*v[3])
rng=np.random.default_rng(1527);out.mkdir(exist_ok=True)
for name,v in [('accepted',force),('random',rng.normal(size=force.shape)),('localized',np.zeros_like(force))]:
    if name=='localized':v[176,0]=1;v[187,0]=-1;v[n//2+176,2]=.5;v[n//2+187,2]=-.5
    f=out/name;f.mkdir(exist_ok=True);np.column_stack((v,np.zeros(n))).astype('f4').tofile(f/'rhs.bin')
