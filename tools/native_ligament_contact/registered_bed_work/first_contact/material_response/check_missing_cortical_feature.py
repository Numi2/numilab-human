"""Remove the newly identified scapular edge and require the GPU to reject the four obstructed poses."""
from pathlib import Path
import json,sys,subprocess
import numpy as np
from compile_connected_joint import digest

def run(source,lib,host,out):
    out.mkdir()
    for filename in ('fibers.bin','poses.bin','edges.bin','dimensions.bin'):
        (out/filename).hardlink_to(source/filename)
    a=np.fromfile(source/'ranges.bin','<u4').reshape(-1,4);a[:,2:]=0;a.tofile(out/'ranges.bin')
    dest=out/'force.bin';p=subprocess.run([host,lib,out,dest],check=True,text=True,capture_output=True)
    z=np.fromfile(dest,'<f4').reshape(-1,3,4);ids=np.fromfile(str(dest)+'.features.bin','<u4')
    q=json.loads((source/'reference.json').read_text())['records'];expected=[i for i,r in enumerate(q) if r['expected_range'] not in set(a[:,0])]
    failed=np.flatnonzero(~np.isfinite(z).all(axis=(1,2))).tolist()
    if failed!=expected or np.flatnonzero(ids==0xffffffff).tolist()!=expected or len(expected)!=4:raise ValueError(f'Incorrect failure set: {failed} vs {expected}')
    result=dict(native=p.stdout.strip(),library=digest(lib),rejected=[q[i]['probe'] for i in expected],valid=len(q)-len(failed),total=len(q));(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*map(lambda x:Path(x).resolve(),sys.argv[1:]))
