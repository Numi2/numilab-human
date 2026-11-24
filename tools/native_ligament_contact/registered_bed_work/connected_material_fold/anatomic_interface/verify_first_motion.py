"""Audit the accepted GPU material coordinates and retain the actual terminal evidence."""
from pathlib import Path
import csv,json,re,sys
import numpy as np
from audit_material_walls import run as audit
from compile_connected_joint import digest
from supported_material_admission import geom

def run(work,capture):
    study=work/'witness-material';v=study/'volume';e=json.loads((capture/'launch.json').read_bytes())
    original=v/'current.node';actual=capture/'physical-accepted.node'
    if not actual.exists():raise ValueError('Native terminal physical capture unavailable')
    with (capture/'native-traces/resting-coupled.csv').open() as f:trace=list(csv.DictReader(f))
    if not trace or int(trace[-1]['step'])!=1 or e['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Wrong coupled release')
    get=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    reference=get(v/'mesh.node');x=get(original);y=get(actual)
    if len(y)!=len(x) or not np.isfinite(y).all()or np.array_equal(x,y):raise ValueError('Invalid accepted-state capture')
    t=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');wall=np.load(v/'material-walls.npz')
    groups=[wall[r] for r in 'OIDC'];base=next(p for p in work.parents if p.name=='human-integrated-gaps-20261104')
    lib=base/'resting-clearance-and-atrium-017/extended-equilibrium-027/connected-arm-volume-043/boundary_pairs.dylib'
    dst=study/'motion-walls.json'
    audit(y,groups,dst,lib)
    rr,_=geom(reference,t[:,:4]-1);roles={};J={};new=np.load(v/'source-overlay.npz')['new_cells']
    for mat,role in [(15,'skin'),(16,'fat')]:
        sel=t[:,4]==mat;vv,ss=geom(y,t[sel,:4]-1);volratio=vv/rr[sel];disp=np.linalg.norm(y-x,axis=1)[np.unique(t[sel,:4]-1)]
        roles[role]=dict(inverted=int((vv<=0).sum()),below_volume=int((vv<=1e-18).sum()),below_shape=int((ss<1e-5).sum()),max_displacement_m=float(max(disp)),min_J=float(min(volratio)),max_J=float(max(volratio)))
    piecevol,_=geom(y,t[new,:4]-1)
    g=json.loads(dst.read_bytes())
    r=dict(launch=digest(capture/'launch.json'),file=digest(actual),initial_file=digest(original),run_exit=json.loads((capture/'execution.json').read_bytes())['exit'],accepted_s=float(trace[-1]['time_s']),roles=roles,replaced_cavity_J=[float(min(piecevol/rr[new])),float(max(piecevol/rr[new]))],surface_counts=g['pair_counts'],surface_intersections=sum(map(sum,g['pair_counts'])),scope='Exact accepted physical coordinates of the four material surfaces at step one. Resting initialization and one endpoint do not certify an arbitrary swept path or all other organs and rigid bodies.')
    (study/'accepted-motion.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent,Path(sys.argv[1]).resolve())
