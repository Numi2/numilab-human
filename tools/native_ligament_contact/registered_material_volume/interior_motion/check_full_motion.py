"""Measure the exact accepted GPU volume and native surface predicates for the full material scene."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from supported_material_admission import geom
from compile_connected_joint import digest
from audit_material_walls import run as walls

def run(case,stage,output):
    mat=stage/'witness-material';v=mat/'volume';arc=case/'native-traces';end=case/'physical-accepted.node'
    trace=list(csv.DictReader((arc/'resting-coupled.csv').open()))
    if not trace:raise ValueError('Native frame never committed')
    i=int(trace[-1]['step'])
    receipt=arc/f'accepted-geometry/step-{i}.receipt.json'
    f=json.loads(receipt.read_bytes());cfg=json.loads((case/'launch.json').read_bytes())
    if f['physical_endpoint']!='accepted' or f['accepted_step']!=i or abs(f['accepted_time_s']-float(trace[-1]['time_s']))>1e-11:raise ValueError('Inconsistent native physical clocks')
    if cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Missing pressure feedback')
    files=[v/'mesh.node',v/'current.node',end]
    x,y,z=[np.loadtxt(k,skiprows=1,usecols=(1,2,3)) for k in files]
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
    if len(x)!=len(y) or not np.isfinite(z).all() or x.shape!=z.shape:raise ValueError('Invalid material frame')
    roles={}
    for role,number in [('skin',15),('fat',16)]:
        t=raw[raw[:,4]==number,:4]-1
        vr,s=geom(x,t);vv,q=geom(z,t);ids=np.flatnonzero(q<1e-5);globalids=np.flatnonzero(raw[:,4]==number)
        roles[role]=dict(inverted=int((vv<=0).sum()),below_volume=int((vv<=1e-18).sum()),below_shape=len(ids),failing_cells=globalids[ids].tolist(),
                         min_J=float(min(vv/vr)),max_J=float(max(vv/vr)),max_displacement_m=float(np.linalg.norm(z-y,axis=1)[np.unique(t)].max()))
    material=np.load(v/'material-walls.npz')
    root=next(p for p in case.parents if p.name=='human-integrated-gaps-20261104')
    lib=root/'resting-clearance-and-atrium-017/extended-equilibrium-027/connected-arm-volume-043/boundary_pairs.dylib'
    wallout=case/'wall-crossings.json';walls(z,[material[k] for k in 'OIDC'],wallout,lib)
    wall=json.loads(wallout.read_bytes())
    target=json.loads((v/'result.json').read_bytes())['updated_sites']
    cids=set(a for r in json.loads((v/'source-overlay.npz').with_name('result.json').read_bytes())['updated_sites'] for a in [r['vertex']])
    affected=np.flatnonzero(np.isin(raw[:,:4]-1,list(cids)).any(axis=1))
    p=dict(accepted_step=i,accepted_s=f['accepted_time_s'],reference=digest(files[0]),loaded=digest(files[1]),physical_frame=digest(end),full_scene_launch=digest(case/'launch.json'),
           receipt_sha256=digest(receipt),world_pose_sha256=f['pack_file_sha256'],execution=json.loads((case/'execution.json').read_bytes()),roles=roles,
           exact_material_surface_pair_count=sum(map(sum,wall['pair_counts'])),wall_pair_counts=wall['pair_counts'],
           defects_in_changed_star={k:sorted(set(r['failing_cells'])&set(affected)) for k,r in roles.items()},
           scope='All skin/fat cells and four registered material surfaces at the accepted native endpoint.')
    output.write_text(json.dumps(p,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('case','stage','output'):p.add_argument(k,type=lambda x:Path(x).resolve())
    run(**vars(p.parse_args()))
