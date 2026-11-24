"""Reintegrate P1 support on the changed cells and replay immutable source mass credits."""
import json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
def run(parent,stage,fit):
    vol=fit/'volume';receipt=json.loads((fit/'result.json').read_text())
    prior=parent/'witness-material'; old=prior/'volume'
    if any(digest(old/k)!=digest(vol/k) for k in ('mesh.ele','cell-mass.f64','node-mass.f64','current-child.labels.csv','material-walls.npz')):raise ValueError('Source mass or topology changed')
    if digest(old/'mesh.node')!=receipt['parent_reference'] or digest(vol/'mesh.node')!=receipt['reference']:raise ValueError('Registration does not match')
    cells=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype=int)-1
    ids=np.flatnonzero(np.isin(cells,receipt['modified_nodes']).any(1))
    if len(ids)!=receipt['local_cells']:raise ValueError('Incomplete cell set')
    np.savez(vol/'source-overlay.npz',old_cells=ids,new_cells=ids,from_cell=ids,to_cell=ids,fraction=np.ones(len(ids)))
    cert=dict(exit=0,native=receipt['importer_result'],source_mesh=receipt['topology'],updated_sites=[dict(vertex=n)for n in receipt['modified_nodes']],inputs={k:digest(vol/k) for k in ('mesh.node','current.node','mesh.ele','cell-mass.f64','node-mass.f64','current-child.labels.csv','material-walls.npz','source-overlay.npz')})
    (vol/'result.json').write_text(json.dumps(cert,indent=2)+'\n')
    out=stage/'witness-material';out.mkdir(exist_ok=True,parents=True)
    for dest,src in [(out/'volume',vol),(stage/'owner.locator',parent/'owner.locator'),(stage/'native-integrated-01',parent/'native-integrated-01'),(stage/'current-p1',prior/'current-p1'),(stage/'recorded-mass',prior/'recorded-mass')]:
        if not dest.exists():dest.symlink_to(src.resolve())
    for name,log in [('rebuild_current_material_basis.py','basis'),('replay_native_credits.py','mass')]:
        dst=stage/name;s=(parent/name).read_text()
        if name.startswith('replay'):
            s,n=re.subn(r"    oldvol=Path\('[^']+'\) if refined", "    oldvol=(work/'current-p1').resolve().parent/'volume' if refined",s)
            if n!=1:raise ValueError('Unrecognized allocator inputs')
        dst.write_text(s)
        with (stage/(log+'.log')).open('w')as f:subprocess.run([sys.executable,dst,'--motion-witness'],check=True,stdout=f,stderr=subprocess.STDOUT)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
