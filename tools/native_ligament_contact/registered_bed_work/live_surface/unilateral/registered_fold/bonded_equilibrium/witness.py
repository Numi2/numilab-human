"""Identify the source and conditioning of the registered fat element at native release."""
from pathlib import Path
import json, sys
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha

def geometry(points):
    e=points[1:]-points[:1]
    determinant=float(np.linalg.det(e))
    h=[]
    for i in range(4):
        a,b,c=np.delete(points,i,axis=0)
        h.append(abs(determinant)/np.linalg.norm(np.cross(b-a,c-a)))
    return dict(volume_m3=determinant/6,altitudes_m=list(h),edge_condition=float(np.linalg.cond(e)))

def measure(root):
    case=root/'bonded-equilibrium'; original=root/'fold-repair/accepted-2'
    evidence=json.loads((original/'verified-endpoint.json').read_text())
    X,Y,T,_,pins=inputs(original)
    result=json.loads((root/'material-direction/release/result.json').read_text())
    row=result['source_cell'];ids=T[row].astype(int); initial=Y[ids].astype(float)
    points={'reference':X[ids].astype(float),'old_start':initial}
    paths={'old_step2':original/'physical-accepted.node','fit_start':case/'current.node',
           'fit_step2':case/'bed-stop/paired/accepted-2/physical-accepted.node'}
    fit=json.loads((case/'fit.json').read_text());prep=json.loads((case/'preparation.json').read_text())
    if pins!=fit['source'] or row!=fit['witness_source_row']:raise ValueError('Different witness or material')
    env=json.loads((original/'launch.json').read_text())['environment']
    with np.load(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as c:
        if not np.any(c['retained_input_cell_rows']==row):raise ValueError('Witness was removed during compilation')
        protected=np.unique(np.r_[c['replacement_tetrahedra'].ravel(),T[np.setdiff1d(np.arange(len(T)),c['retained_input_cell_rows'])].ravel()])
    expected={'old_step2':evidence['physical_nodes_sha256'],'fit_start':prep['loaded_pose']}
    ep=case/'bed-stop/paired/accepted-2/verified-endpoint.json'
    if ep.exists():expected['fit_step2']=json.loads(ep.read_text())['physical_nodes_sha256']
    for k,h in expected.items():
        p=paths[k]
        if sha(p)!=h:raise ValueError('Changed '+k)
        points[k]=np.loadtxt(p,skiprows=1,usecols=(1,2,3))[ids]
    with np.load(case/'preload.npz') as v:moved=set(map(int,v['rows']))
    out={'source_cell':row,'native_cell':result['native_cell'],'nodes':ids.tolist(),
         'reference_sha256':pins['reference_sha256'],'vertices_protected_by_previous_remesh':np.isin(ids,protected).tolist(),
         'vertex_received_fit':[int(i) in moved for i in ids],'snapshots':{},'pins':expected}
    for name,p in points.items():
        out['snapshots'][name]=dict(**geometry(p),displacement_from_old_start_m=np.linalg.norm(p-initial,axis=1).tolist())
    return out
if __name__=='__main__':print(json.dumps(measure(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))
