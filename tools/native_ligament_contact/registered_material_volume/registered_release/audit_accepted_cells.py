"""Trace every low-quality accepted material cell to its source dimensions, strain and geometric constraints."""
from pathlib import Path
import json,re,sys
import numpy as np
from supported_material_admission import geom
from compile_connected_joint import digest

def run(vol,pose,case,motion,ports,bindings,out):
    m=json.loads(motion.read_bytes());trace=json.loads((case/'launch.json').read_bytes())
    for p,h in [(vol/'mesh.node',m['reference']),(pose,m['loaded']),(case/'physical-accepted.node',m['physical_frame']),(case/'launch.json',m['full_scene_launch'])]:
        if digest(p)!=h:raise ValueError('Mismatched physical frame')
    table=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    labels=np.loadtxt(vol/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')
    if not np.array_equal(labels[:,0],np.arange(len(table))) or not np.array_equal(labels[:,1],table[:,4]):raise ValueError('Invalid source identities')
    t=table[:,:4]-1
    xyz=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in (vol/'mesh.node',pose,case/'physical-accepted.node')]
    gm=[geom(x,t) for x in xyz]
    report=set(i for r in m['roles'].values() for i in r['failing_cells'])
    idx=np.flatnonzero(gm[2][1]<1e-5)
    if set(idx)!=report:raise ValueError('Different shape witnesses')
    walls=np.load(vol/'material-walls.npz');z=np.load(ports/'material-contact.npz')
    membership={k:np.unique(walls[k]) for k in 'OIDC'}
    attachment=np.asarray([int(u)for u in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',bindings.read_text(),re.M)])
    if not attachment.size:raise ValueError('Missing loaded interface bindings')
    protected=np.unique(np.r_[*[membership[k]for k in 'ODC'],attachment,z['facets'][:,4:7].ravel(),z['samples'][:,0].astype('i4')])
    edges=[x[t[idx,1:]]-x[t[idx,:1]]for x in xyz]
    sv=np.linalg.svd(np.linalg.solve(edges[0],edges[2]),compute_uv=False)
    nodes=np.unique(t[idx]);mat=table[:,4]
    same=np.array([len(np.unique(mat[(t==node).any(1)]))==1 for node in nodes])
    homogeneous=nodes[same];can=np.isin(t[idx],protected,invert=True)
    rows=[dict(cell=int(i),material=int(mat[i]),donor=int(labels[i,2]),region=int(labels[i,3]),face=int(labels[i,4]),corners=t[i].tolist(),free_corners=t[i][can[j]].tolist(),homogeneous_free_corners=np.intersect1d(t[i][can[j]],homogeneous).tolist(),shape=[float(v[1][i])for v in gm],J=[float(v[0][i]/gm[0][0][i])for v in gm],accepted_stretches=sv[j].tolist())for j,i in enumerate(idx)]
    summary=dict(cells=len(t),source_mass_sha256=digest(vol/'node-mass.f64'),geometry_sha256=m['physical_frame'],shapes_below=[int(np.sum(q<1e-5))for v,q in gm],witness_cells=len(idx),with_unprotected_corner=int(can.any(1).sum()),with_homogeneous_unprotected_corner=sum(bool(r['homogeneous_free_corners']) for r in rows),source_shape_lt_twice_gate=sum(r['shape'][0]<2e-5 for r in rows),source_shape_lt_five_times_gate=sum(r['shape'][0]<5e-5 for r in rows),source_donors=sorted(set(r['donor'] for r in rows)),rows=rows)
    out.write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items()if k not in ('source_donors','rows')})
if __name__=='__main__':run(*(Path(p).resolve()for p in sys.argv[1:]))
