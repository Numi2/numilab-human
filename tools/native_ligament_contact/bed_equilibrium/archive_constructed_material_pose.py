"""Save constructed current coordinates in the existing TetGen node format and prove recovery."""
import argparse,gzip,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from evaluate_registered_bed_transfer import digest
from screen_registered_dense_posture import load_geometry
from recover_material_thickness import audit
from untangle_registered_donor_field import differentials

def run(full,donor,pose,label):
    x,_,_,t,_,_,_,pins=load_geometry(full,donor)
    q=x+np.load(pose/'displacement.npy')
    rr=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    if not np.array_equal(q,q.astype('f4').astype(float)) or audit(q,t,rr)[3].any():raise ValueError('unqualified emitted geometry')
    if json.loads((pose/'loaded-registration.json').read_text())['source']!=pins:raise ValueError('unmatched construction')
    tag='human-closed-skin-diagnostic-packs-20261121';repo='Numi2/numilab-human'
    with tempfile.TemporaryDirectory(dir=pose) as scratch:
        temp=Path(scratch);node=temp/'current.node'
        with node.open('w') as f:
            f.write(f'{len(q)} 3 0 0\n')
            for i,row in enumerate(q,1):f.write(f'{i} {row[0]:.17g} {row[1]:.17g} {row[2]:.17g}\n')
        raw=digest(node);name='registered-loaded-skin-fat-'+label+'-'+raw[:16]+'.node.zst';asset=temp/name
        subprocess.run(['/opt/homebrew/bin/zstd','-q','-8',str(node),'-o',str(asset)],check=True)
        gh=['/opt/homebrew/bin/gh','release']
        assets=json.loads(subprocess.check_output(gh+['view',tag,'-R',repo,'--json','assets'],text=True))['assets']
        if name not in {v['name'] for v in assets}:
            subprocess.run(gh+['upload',tag,str(asset),'-R',repo],check=True)
        dst=temp/'download';dst.mkdir()
        subprocess.run(gh+['download',tag,'-R',repo,'-p',name,'-D',str(dst)],check=True)
        if digest(asset)!=digest(dst/name):raise ValueError('download identity mismatch')
        out=dst/'restored.node';subprocess.run(['/opt/homebrew/bin/zstd','-q','-d',str(dst/name),'-o',str(out)],check=True)
        rows=np.loadtxt(out,skiprows=1);assert np.array_equal(rows[:,0],np.arange(1,len(q)+1)) and np.array_equal(rows[:,1:],q)
        if digest(out)!=raw:raise ValueError('restored TetGen data mismatch')
        recovery=dict(repository=repo,tag=tag,asset=name,bytes=asset.stat().st_size,compressed_sha256=digest(asset),decoded_node_sha256=raw,
                     exact_recovered_coordinates=len(q),source=pins,url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                     role='constructed current positions; conserved rest mesh and masses remain in native owner')
        (pose/'recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    print(raw,len(q),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('full','donor','pose'):p.add_argument(key,type=Path)
    p.add_argument('label');run(**vars(p.parse_args()))
