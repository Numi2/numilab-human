"""Measure the installed pose relative to its actual material, with reference-volume weighting."""
from pathlib import Path
import json,sys
import numpy as np
from compile_connected_joint import digest

def run(root,out):
    verified=json.loads((root/'result.json').read_bytes())
    rest,init,ele=(root/file for file in ('mesh.node','current.node','mesh.ele'))
    for path,k in [(rest,'material'),(init,'initial'),(ele,'unchanged_topology')]:
        if digest(path)!=verified[k]:raise ValueError('The registered mesh changed')
    x,y=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8') for p in [rest,init]]
    rows=np.loadtxt(ele,skiprows=1,usecols=(1,2,3,4,5),dtype='i4');n=len(rows)
    v=np.empty(n);jac=np.empty(n);maxstretch=np.empty(n);minstretch=np.empty(n)
    for i in range(0,n,16384):
        sl=slice(i,min(i+16384,n));ids=rows[sl,:4]-1
        a=x[ids[:,1:]]-x[ids[:,:1]];b=y[ids[:,1:]]-y[ids[:,:1]]
        v[sl]=np.linalg.det(a)/6;jac[sl]=np.linalg.det(b)/np.linalg.det(a)
        s=np.linalg.svd(np.linalg.solve(a,b),compute_uv=False);maxstretch[sl]=s[:,0];minstretch[sl]=s[:,-1]
    if (v<=0).any() or (jac<=0).any() or not np.all(np.isfinite(maxstretch)):raise ValueError('Invalid mesh')
    stats={}
    for m in [15,16]:
        it=rows[:,4]==m;w=v[it];j=jac[it];hi=maxstretch[it];lo=minstretch[it]
        measures={}
        for name,sel in [('delta_J_gt_10_percent',np.abs(j-1)>.1),('delta_J_gt_50_percent',np.abs(j-1)>.5),('max_principal_stretch_gt_2',hi>2),('min_principal_stretch_lt_half',lo<.5)]:
            measures[name]=dict(cells=int(sel.sum()),reference_volume_fraction=float(w[sel].sum()/w.sum()))
        stats[m]=dict(cells=int(it.sum()),reference_volume_m3=float(w.sum()),volume_ratio=float(w@j/w.sum()),max_principal_stretch=float(np.max(hi)),min_principal_stretch=float(np.min(lo)),J_min=float(j.min()),J_max=float(j.max()),descriptive_thresholds=measures)
    out.write_text(json.dumps(dict(rest=verified['material'],initial=verified['initial'],elements=verified['unchanged_topology'],layers=stats,scope='Initial registered geometry. Principal stretches use the rest-to-loaded affine mapping for each tetrahedron. Thresholds describe deformation and do not establish material injury or a constitutive failure.'),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
