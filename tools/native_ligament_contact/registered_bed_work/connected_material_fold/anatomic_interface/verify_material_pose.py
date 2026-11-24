from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,ctypes,numpy as np,sys
from compile_connected_joint import digest
from measure_bone_tissue_overlap import run as bones
from audit_material_walls import run as walls

def run(work,lib,field,out):
    out.mkdir();q=work.parent;j=q.parent.parent;v=q/'qualified-material';x=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3));y=x.copy();z=np.load(field);y[z['nodes']]=z['positions']
    t=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    checker=Path((j/'storage/interval-checker-locator.txt').read_text().split()[1]);fn=ctypes.CDLL(str(checker)).scanCellIntervals;dp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS');fn.argtypes=[dp,dp,ctypes.c_size_t,ip,ctypes.c_size_t,dp];vals=np.empty((len(t),3))
    if fn(x,y,len(y),np.ascontiguousarray(t,dtype='i4'),len(t),vals):raise ValueError('Interval evaluation failed')
    mar=np.minimum(vals[:,0]-vals[:,1]-float(np.float32(.35)),2.5-vals[:,0]-vals[:,1])
    ws=np.load(v/'material-walls.npz');G=[ws[s]for s in 'OIDC']
    with ThreadPoolExecutor(2) as pool:
        results=[pool.submit(bones,j.parent/'ligament-equilibrium-070',out/'bones',j/'distal-registered-native/target-pose.json',field,v),pool.submit(walls,y,G,out/'walls.json',lib)]
        for r in results:r.result()
    old=json.loads((v/'bone-overlap/result.json').read_bytes())['results'];new=json.loads((out/'bones/result.json').read_bytes())['results']
    if set(old)!=set(new):raise ValueError('Incomplete physical bone coverage')
    diff={k:{r:float(row[r]-old[k][r]) for r in ('skin','fat')}for k,row in new.items()};chg={k:d for k,d in diff.items()if max(abs(v)for v in d.values())>1e-9};bad=np.flatnonzero(~(mar>0));ws=json.loads((out/'walls.json').read_bytes())
    result=dict(base=digest(v/'result.json'),field=digest(field),interval_violations=len(bad),worst_interval_margin=float(mar.min()),violating_cells=bad[:64].tolist(),wall_crossings=sum(map(sum,ws['pair_counts'])),changed_rigid_overlap_mm3=chg,total_change_mm3={s:sum(row[s]for row in diff.values())for s in ('skin','fat')})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':run(Path(__file__).resolve().parent,Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
