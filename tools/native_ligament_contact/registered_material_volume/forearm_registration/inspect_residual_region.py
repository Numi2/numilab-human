"""Cross-reference a native residual peak with actual incident tissue, wall membership and elastic loads."""
import sys,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from prepare_indentation import soft_response
def run(v,motion,fit,material,out):
    out.mkdir(exist_ok=True);witness=json.loads(Path('remaining-load.json').read_bytes());peak=witness['skin_fat_peak_node_zero_based']
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    base=np.asarray(witness['incident_cells']);T=t[base]
    if not np.array_equal(base,np.flatnonzero(np.isin(t,peak).any(1))):raise ValueError('Stale force witness')
    walls=np.load(v/'material-walls.npz');pts=[np.loadtxt(p,skiprows=1,usecols=(1,2,3))for p in (v/'mesh.node',fit/'current.node',motion/'physical-accepted.node')]
    x,y,z=pts;coeff=json.loads(material.read_bytes())['materials'];params=np.array([[coeff[str(m)]['registered'][k]for k in ['source_mu_pa','source_logJ_coefficient_pa']]for m in raw[base,4]]).T
    force=soft_response(x,y,T,*params);stats={}
    for g,P in [('initial',y),('first_accepted',z)]:
        f,j=force(P);i,k=np.nonzero(T==peak);loads=f[i,k];net=loads.sum(0);along=loads@net/np.linalg.norm(net);order=np.argsort(abs(along))[::-1]
        stats[g]=dict(net_N=np.linalg.norm(net),cells=[dict(row=int(base[i]),material=int(raw[base[i],4]),signed_N=along[i],J=j[i],vertices=t[base[i]].tolist())for i in order[:4]],largest_J_cell=int(base[np.argmax(j)]),largest_J=float(max(j)),smallest_J_cell=int(base[np.argmin(j)]),smallest_J=float(min(j)))
    d=dict(peak=peak,wall_membership=[s for s in 'OIDC' if np.any(walls[s]==peak)],native=digest(Path('remaining-load.json')),stages=stats)
    (out/'region.json').write_text(json.dumps(d,indent=2)+'\n');print(d)
if __name__=='__main__':run(*(Path(v).resolve()for v in sys.argv[1:]))
