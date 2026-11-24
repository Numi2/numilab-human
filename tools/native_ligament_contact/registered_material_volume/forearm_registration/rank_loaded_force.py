"""Attribute the assembled forearm internal force to the individual registered skin and fat cells."""
import json,sys
from pathlib import Path
import numpy as np
from prepare_indentation import soft_response

def rank(volume, material, fit):
    a=np.load('cavity.npz');ids=a['ids'];x,y,z=a['xyz'];nodes=a['local'][np.flatnonzero(a['rows']==a['selected'])[0]]
    raw=json.loads(material.read_bytes())['materials'];rows=np.searchsorted(a['rows'],a['neighbors']);t=a['local'][rows];tags=a['labels'][rows,1]
    mu,bulk=np.array([[raw[str(m)]['registered'][k] for k in ('source_mu_pa','source_logJ_coefficient_pa')]for m in tags]).T
    response=soft_response(x,y,t,mu,bulk);ref=x[t[:,1:]]-x[t[:,:1]];V=np.linalg.det(ref)/6
    allP=np.loadtxt(fit/'current.node',skiprows=1,usecols=(1,2,3))[ids]
    records={}
    for key,p in [('initial',y),('first_accepted',z),('candidate',allP)]:
        F=np.linalg.solve(ref,p[t[:,1:]]-p[t[:,:1]]);forces,J=response(p)
        per=forces*np.isin(t,nodes)[:,:,None];net=np.zeros_like(x);np.add.at(net,t.ravel(),per.reshape(-1,3))
        peak=nodes[np.argmax(np.linalg.norm(net[nodes],axis=1))];k,l=np.nonzero(t==peak);direction=net[peak]/np.linalg.norm(net[peak])
        take=np.argsort(abs(forces[k,l]@direction))[::-1][:5]
        records[key]=dict(node=int(ids[peak]),force_N=np.linalg.norm(net[peak]),cells=[dict(row=int(a['neighbors'][i]),role=int(tags[i]),along_net_N=float(forces[i,j]@direction),J=J[i],stretch=np.linalg.svd(F[i],compute_uv=False).tolist(),vol_m3=V[i]) for i,j in zip(k[take],l[take])])
    Path('force-owners.json').write_text(json.dumps(records,indent=2)+'\n');print({k:[(v['row'],v['role'],round(v['along_net_N'],1),np.round(v['stretch'],1))for v in r['cells'][:2]]for k,r in records.items()})
if __name__=='__main__':rank(*(Path(v)for v in sys.argv[1:]))
