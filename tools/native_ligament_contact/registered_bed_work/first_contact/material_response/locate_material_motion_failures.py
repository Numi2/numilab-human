"""Record exact physical cells that fail the existing supported-material motion audit."""
import json,sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from supported_material_admission import material_state,geom
from register_material_surfaces import sha


def run(trial,mesh,out):
    cfg=json.loads((trial/'launch.json').read_bytes())
    e=cfg['environment']
    x,y,t,mat,_=inputs(trial)
    if 'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ' in e or sha(mesh/'mesh.ele')!=e['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256']:
        raise ValueError('Physical endpoint does not refer to this material topology')
    p=trial/'physical-accepted.node';loaded=np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    props,_=materials(Path(cfg['compiled_material_owner']))
    origin=min(r['first_cell'] for r in json.loads((mesh/'receipt.json').read_bytes())['regions'])
    rows=[]
    for pos in range(0,len(t),32768):
        s=slice(pos,pos+32768);cells=t[s]
        X=x[cells].astype(float);Y=loaded[cells]
        J,L,P=material_state(X,Y,props[mat[s]-15])
        vol,shape=geom(loaded,cells)
        geom_bad=(vol<=1e-18)|(shape<1e-5)|(J<.4)|(J>2.5)
        fail=(L[:,-1]<2/3)|(L[:,0]>1.5)|geom_bad
        for k in np.flatnonzero(fail):
            i=pos+k;T=cells[k]
            edge=X[k,1:]-X[k,:1]
            rows.append(dict(cell=int(i),material=int(mat[i]),rebuilt_region=bool(i>=origin),nodes=T.tolist(),
                source_shortest_matrix_singular_value_m=float(np.linalg.svd(edge,compute_uv=False)[-1]),
                reference_volume_m3=float(np.linalg.det(edge)/6),current_volume_m3=float(vol[k]),current_shape=float(shape[k]),
                J=float(J[k]),principal_stretches=L[k].tolist(),max_node_motion_m=float(np.linalg.norm(Y[k]-X[k],axis=1).max()),
                geometry_failed=bool(geom_bad[k])))
    out.write_text(json.dumps(dict(accepted_pose=sha(p),cells_audited=len(t),failures=rows),indent=2)+'\n')
    print([(r['cell'],r['material'],r['geometry_failed'],round(r['J'],4)) for r in rows])


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
