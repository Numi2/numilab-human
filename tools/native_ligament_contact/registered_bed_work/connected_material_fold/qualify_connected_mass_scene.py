"""Import the exact full physical volume with native density, interval and four-wall checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,ctypes,os,subprocess
import numpy as np
from audit_material_walls import run as audit
from measure_density_capacity import density
from compile_connected_joint import digest

def run(q):
    j=q.parent.parent;w=q/'remaining-fold';v=w/'collar-2/fold-volume';rec=q/'local-density-transfer'
    out=q/'qualified-material';out.mkdir(exist_ok=True)
    z=np.load(rec/'allocation.npz');doc=json.loads((rec/'result.json').read_bytes())
    if digest(rec/'allocation.npz')!=doc['candidate'] or digest(v/'result.json')!=doc['source']:raise ValueError('Unbound candidate')
    x,t,l,m,_,_=density(v);y=np.loadtxt(v/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    use=z['nodes'];row=z['cell_rows'];src=z['credit_from'];dst=z['credit_to'];credit=z['credit_kg']
    if not np.array_equal(x[use],z['original_positions']) or not np.array_equal(y[use],x[use]) or not np.array_equal(m[row],z['mass_before']):raise ValueError('Prior positions or masses changed')
    if not np.all(l[src,1:]==l[dst,1:]) or any(len(set(t[a])&set(t[b]))!=3 for a,b in zip(src,dst)) or min(credit)<=0:raise ValueError('Invalid same-source shared-face transfer')
    expected=m.copy();np.add.at(expected,src,-credit);np.add.at(expected,dst,credit)
    x[use]=z['positions'];y[use]=z['positions'];m[row]=z['mass_after']
    if np.max(abs(m-expected))>1e-17:raise ValueError('Unexplained redistribution')
    for f in ('mesh.ele','current-child.labels.csv'):
        if not (out/f).exists():os.link(v/f,out/f)
    for f,xyz in [('mesh.node',x),('current.node',y)]:
        with (out/f).open('w')as stream:
            stream.write(f'{len(xyz)} 3 0 0\n');np.savetxt(stream,np.c_[np.arange(1,len(xyz)+1),xyz],fmt=['%d','%.17g','%.17g','%.17g'])
    m.astype('<f8').tofile(out/'cell-mass.f64');nm=sum(np.bincount(c,weights=m/4,minlength=len(x))for c in t.T);nm.astype('<f8').tofile(out/'node-mass.f64')
    u=np.flatnonzero(np.any(x!=y,axis=1));np.savez_compressed(out/'actual-transport.npz',nodes=u,positions=y[u].astype('f4'))
    checker=Path((j/'storage/interval-checker-locator.txt').read_text().split()[1]);proof=json.loads((j/'verified-connected-material/interval-audit.json').read_bytes())
    if digest(checker)!=proof['checker']:raise ValueError('Unpinned determinant checker')
    dp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(checker)).scanCellIntervals;fn.argtypes=[dp,dp,ctypes.c_size_t,ip,ctypes.c_size_t,dp];vals=np.empty((len(t),3))
    if fn(x,y,len(x),np.ascontiguousarray(t),len(t),vals):raise ValueError('Interval computation failed')
    mar=np.minimum(vals[:,0]-vals[:,1]-float(np.float32(.35)),2.5-vals[:,0]-vals[:,1])
    if not np.all(mar>0):raise ValueError('Invalid cell interval')
    cmd=json.loads((v/'native-launch.json').read_bytes())['command'];cmd[1]=str(out)
    with (out/'native-import.json').open('w')as stdout,(out/'native-import.stderr').open('w')as stderr:
        code=subprocess.run(cmd,stdout=stdout,stderr=stderr).returncode
    nr=json.loads((out/'native-import.json').read_bytes());meta=dict(exit=code,native_command=cmd,binary=digest(Path(cmd[0])),import_report=nr,interval_failures=int(np.sum(mar<=0)),min_interval_margin=float(min(mar)),source_volume=digest(v/'result.json'),allocation=digest(rec/'allocation.npz'),inputs={f:digest(out/f)for f in ['mesh.ele','mesh.node','current.node','current-child.labels.csv','cell-mass.f64','node-mass.f64']})
    (out/'result.json').write_text(json.dumps(meta,indent=2)+'\n');print('native import',code,nr,flush=True)
    if code:raise ValueError('Native import failed')
    a=np.load(w/'source-surfaces.npz');side=np.load(w/'collar-2/consolidated-fold/surface-ownership.npz');groups=[a[n] for n in 'OIDC'];prev=side['old_roles_rows'];after=side['new_roles_rows_nodes']
    for role in set(prev[:,0]):groups[role]=np.vstack([np.delete(groups[role],prev[prev[:,0]==role,1],axis=0),after[after[:,0]==role,2:]])
    np.savez_compressed(out/'material-walls.npz',**dict(zip('OIDC',groups)))
    base=next(s for s in q.parents if s.name=='human-integrated-gaps-20261104')
    lib=base/'resting-clearance-and-atrium-017/extended-equilibrium-027/connected-arm-volume-043/boundary_pairs.dylib'
    from measure_bone_tissue_overlap import run as bones
    def geometry():
        for name,xyz in [('reference',x),('registered',y)]:audit(xyz,groups,out/(name+'-all-walls.json'),lib)
    with ThreadPoolExecutor(2)as pool:
        fs=[pool.submit(geometry),pool.submit(bones,j.parent/'ligament-equilibrium-070',out/'bone-overlap',j/'distal-registered-native/target-pose.json',out/'actual-transport.npz',out)]
        for f in fs:f.result()
if __name__=='__main__':run(Path(__file__).resolve().parent)
