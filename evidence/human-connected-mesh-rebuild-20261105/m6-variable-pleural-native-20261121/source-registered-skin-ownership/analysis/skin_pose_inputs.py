"""Retain the native outer skin vertices and the exact correspondence of their source faces."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def capture(pack,out,analysis,workspace):
    sys.path[:0]=[str(workspace/'resting-surface-clearance-001/src'),str(workspace/'coupled-crossing-repair-001'),str(workspace/'resting-surface-clearance-001'),str(workspace/'resting-clearance-and-atrium-002/publication/scripts')]
    from bindings import read_skin
    from rebind_human_skin_reference import decode
    r=json.loads(pack.with_suffix('.receipt.json').read_text())
    cfg=json.loads((out/'launch.json').read_text())
    src=Path(cfg['argv'][cfg['argv'].index('--skin-payload')+1])
    if digest(src)!=cfg['pins'][str(src)] or digest(pack)!=r['pack_file_sha256']:raise ValueError('Capture identity mismatch')
    q=decode(src.read_bytes())
    v,f=read_skin(pack,r)
    if not np.array_equal(f,q['inds']):raise ValueError('Detailed face to source ancestry changed')
    sm=r['skin_source_mapping']
    if sm['influence_layout']!='vertex-major':raise ValueError('Unsupported influence-order recording')
    for k in ['influences','vertex_map']:
        if digest(Path(sm[k]['path']))!=sm[k]['sha256']:raise ValueError('Compiled skin binding changed')
    np.savez_compressed(analysis/('skin-pose-'+str(r['accepted_step'])+'.npz'),positions=v,source_sha256=digest(src),pack_sha256=r['pack_file_sha256'],faces_sha256=hashlib.sha256(f.astype('<u4').tobytes()).hexdigest())
if __name__=='__main__':capture(*map(Path,sys.argv[1:]))
