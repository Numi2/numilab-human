"""Close child fractions against double reference volumes exported by the production importer."""
import json,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
source,volfile,mesh,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True,parents=True)
p=source/'current-density-closed-flow.npz';r=json.loads((source/'mass-transfer.json').read_text());assert sha(p)==r['flow_sha256']
f=dict(np.load(p));g=f['group_for_cell'];em=f['element_mass_kg'].copy();v=np.fromfile(volfile,dtype='f8');assert len(v)==len(g) and np.all(v>1e-18)
with (mesh/'mesh.ele').open() as h:n=int(next(h).split()[0]);e=np.loadtxt(h,max_rows=n,usecols=(1,2,3,4,5),dtype='i4')
t=e[:,:4]-1;lo=np.where(e[:,4]==15,900.,850.);hi=np.where(e[:,4]==15,1100.,950.)
gm=np.bincount(f['group'],weights=f['mass_flow_kg']);gv=np.bincount(g,weights=v);average=gm/gv
assert np.all((average[g]>=lo)&(average[g]<=hi))
actual=em*(gm/np.bincount(g,weights=em))[g]/v
cross=(actual<lo)|(actual>hi)|(em/v<lo)|(em/v>hi);groups=np.unique(g[cross])
sel=np.isin(g,groups);em[sel]=v[sel]*average[g[sel]]
actual=em*(gm/np.bincount(g,weights=em))[g]/v
assert np.all((actual>=lo)&(actual<=hi)&(em/v>=lo)&(em/v<=hi))
f['element_mass_kg']=em;f['nodal_mass_kg']=sum(np.bincount(t[:,i],weights=em*.25,minlength=len(f['nodal_mass_kg'])) for i in range(4));dst=out/p.name
np.savez_compressed(dst,**f);r.update(flow_sha256=sha(dst),native_reference_volumes_sha256=sha(volfile),pre_rounding_flow_sha256=sha(p),corrected_recipients=groups.tolist(),corrected_cells=int(sel.sum()),unchanged_carrier_route_masses=True,minimum_native_density_margin_kg_m3=float(np.min(np.minimum(actual-lo,hi-actual))),density_outliers=0,result_mass_kg=float(em.sum()),nodal_mass_kg=float(f['nodal_mass_kg'].sum()))
(out/'mass-transfer.json').write_text(json.dumps(r,indent=2)+'\n');print('groups',len(groups),'cells',int(sel.sum()),'minimum margin',r['minimum_native_density_margin_kg_m3'])
