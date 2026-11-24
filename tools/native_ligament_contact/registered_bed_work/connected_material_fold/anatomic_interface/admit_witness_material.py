"""Transfer material mass by polyhedral intersection and run the native import on the new star."""
import ctypes,json,os,subprocess
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from material_cavity_flips import boundary
from supported_material_admission import geom
from transfer_cavities import intersection_fraction

def run(work):
    q=work.parent;j=q.parent.parent;src=q/'qualified-material';destination=work/'witness-material';out=destination/'volume';out.mkdir(exist_ok=True,parents=True)
    load=lambda p:json.loads(p.read_bytes())
    srcmesh=load(src/'result.json')
    z=np.load(work/'witness-cavity/material-star.npz');rows=z['source_cells'];readxyz=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    if len(z['interior']):raise ValueError('Apex must be a new interior material coordinate')
    x=np.vstack((readxyz(src/'mesh.node'),z['apex_reference']));y=np.vstack((readxyz(work/'selected-pose/current.node'),z['apex_registered']))
    raw=np.loadtxt(src/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    fan=np.c_[np.full(len(z['faces']),len(x)-1,'i4'),z['faces']]
    if boundary(t[rows])!=boundary(fan):raise ValueError('Changed cavity boundary')
    label=np.loadtxt(src/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')
    if not (label[rows,1:]==label[rows[0],1:]).all():raise ValueError('Mixed source identity')
    new=np.r_[rows,np.arange(len(t),len(t)+len(fan)-len(rows))]
    frac=np.array([[intersection_fraction(a,b,x)for b in fan]for a in t[rows]])
    v=geom(x,t[rows])[0];v1=geom(x,fan)[0];dm=np.fromfile(src/'cell-mass.f64','<f8')
    row_error=max(abs(frac.sum(1)-1));col_error=max(abs((frac.T@v)/v1-1))
    if max(row_error,col_error)>3e-9:raise ValueError('Subcell volume closure mismatch '+str((row_error,col_error)))
    frac/=frac.sum(1)[:,None]
    target=frac.T@dm[rows]
    if max((target/v1))>950.0000001 or np.any(target/v1<850.-1e-7):raise ValueError('Unexpected local adipose density range '+str(target/v1))
    tt=np.vstack((t,np.zeros((len(fan)-len(rows),4),dtype='i4')));tt[new]=fan
    material=np.r_[raw[:,4],np.full(len(fan)-len(rows),16)]
    masses=np.r_[dm,np.zeros(len(fan)-len(rows))];masses[new]=target
    nodes=sum(np.bincount(tt[:,k],weights=masses/4,minlength=len(x))for k in range(4))
    if min(nodes)<=0 or abs(target.sum()-dm[rows].sum())>1e-17:raise ValueError('Unowned mass')
    # Preserve previous file payloads outside the altered rows with the same source row identities.
    def append_point(old,p,point):
        with old.open('rb')as inp,p.open('wb')as dst:
            inp.readline();dst.write(f'{len(x)} 3 0 0\n'.encode())
            import shutil
            shutil.copyfileobj(inp,dst);dst.write((str(len(x))+' '+' '.join(format(val,'.17g')for val in point)+'\n').encode())
    append_point(src/'mesh.node',out/'mesh.node',x[-1]);append_point(work/'selected-pose/current.node',out/'current.node',y[-1])
    with (out/'mesh.ele').open('w')as f:
        f.write(f'{len(tt)} 4 1\n');np.savetxt(f,np.column_stack((np.arange(1,len(tt)+1),tt+1,material)),fmt='%d')
    with (src/'current-child.labels.csv').open('rb')as inp,(out/'current-child.labels.csv').open('wb')as dst:
        import shutil
        shutil.copyfileobj(inp,dst)
        for i in range(len(t),len(tt)):dst.write((','.join(map(str,[i,*label[rows[0],1:]]))+'\n').encode())
    masses.astype('<f8').tofile(out/'cell-mass.f64');nodes.astype('<f8').tofile(out/'node-mass.f64')
    dr,dc=np.nonzero(frac>0);np.savez_compressed(out/'source-overlay.npz',from_cell=rows[dr],to_cell=new[dc],fraction=frac[dr,dc],mass_kg=dm[rows[dr]]*frac[dr,dc],old_cells=rows,new_cells=new)
    # Each exterior material triangle has the same positions and oriented connectivity.
    for name in ('material-walls.npz',):
        dest=out/name
        if not dest.exists():os.link(src/name,dest)
    for role,f in np.load(out/'material-walls.npz').items():
        if (f>=len(x)-1).any():raise ValueError('New interior node is on the boundary')
        fs={tuple(sorted(k))for k in boundary(t[rows])}
        inner={tuple(sorted(k))for tet in t[rows] for k in __import__('itertools').combinations(tet,3)}-fs
        if any(tuple(sorted(a)) in inner for a in f[np.isin(f,np.unique(t[rows])).all(1)]):raise ValueError('Material interface inside proposed cavity')
    checker=Path((j/'storage/interval-checker-locator.txt').read_text().split()[1]);proof=load(j/'verified-connected-material/interval-audit.json')
    if digest(checker)!=proof['checker']:raise ValueError('Unpinned geometry audit')
    dp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(checker)).scanCellIntervals
    fn.argtypes=[dp,dp,ctypes.c_size_t,ip,ctypes.c_size_t,dp];vals=np.empty((len(tt),3))
    if fn(x,y,len(x),np.ascontiguousarray(tt),len(tt),vals):raise ValueError('Interval audit failed')
    mar=np.minimum(vals[:,0]-vals[:,1]-float(np.float32(.35)),2.5-vals[:,0]-vals[:,1])
    if np.any(mar<=0):raise ValueError('Uncertified registered interval')
    cmd=load(q/'remaining-fold/collar-2/fold-volume/native-launch.json')['command'];cmd[1]=str(out)
    with (out/'native-import.json').open('w')as so,(out/'native-import.stderr').open('w')as se:
        proc=subprocess.run(cmd,stdout=so,stderr=se)
    result=dict(exit=proc.returncode,inputs={k:digest(out/k) for k in ['mesh.node','current.node','mesh.ele','cell-mass.f64','node-mass.f64','current-child.labels.csv','source-overlay.npz']},source=digest(src/'result.json'),min_interval_margin=float(min(mar)),maximum_source_relative_error=float(row_error),maximum_destination_relative_error=float(col_error),local_density_range_kg_m3=[min(target/v1),max(target/v1)],import_report=load(out/'native-import.json'),total_nodes=len(x),total_cells=len(tt),loaded_walls_preserved=True,original_registered_coordinates=len(x)-1)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if proc.returncode or result['import_report']['density_outliers']:raise ValueError('Native material admission failed')
if __name__=='__main__':run(Path(__file__).resolve().parent)
