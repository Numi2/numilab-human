"""Recover a joined material candidate from its XOR delta and recheck native reference results."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from transfer_connected_volume import geometry

def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def run(material,trial,native,results,out):
    data=np.load(material);x=np.ascontiguousarray(data['x']);y=np.ascontiguousarray(np.load(trial))
    if y.dtype!=np.float32 or x.dtype!=np.float32:raise ValueError('Require emitted native Float32')
    a,b=x.view('u4'),y.view('u4');ix=np.flatnonzero(np.any(a!=b,axis=1))
    np.savez_compressed(out/'coordinate-delta.npz',rows=ix.astype('u4'),xor=a[ix]^b[ix])
    with np.load(out/'coordinate-delta.npz') as patch:
        copy=x.copy();copy.view('u4')[patch['rows']]^=patch['xor']
        if not np.array_equal(copy.view('u4'),b):raise ValueError('Recovered coordinates differ')
    t=data['tets'];types=data['tags'];v,q=geometry(y,t);ref=np.fromfile(native,'f8')
    if not np.allclose(v,ref,rtol=2e-7,atol=1e-24):raise ValueError('Native/authoring volume discrepancy')
    current=json.loads((results/'contact-delta.json').read_text())
    def count(rows):return sum(len(p) for p in rows.values())
    layers={}
    for k in ('introduced','resolved','persistent'):
        groups=current[k];layers[k]={f'{l}/{r}':sum(len(v) for name,v in groups.items() if name.startswith(l+'/') and (name.split('/')[1].startswith('bone_') if r=='bone' else not name.split('/')[1].startswith('bone_'))) for l in ('outer','inner','deep') for r in ('bone','layer')}
    assert current['baseline']==count(current['resolved'])+count(current['persistent'])
    assert current['proposed']==count(current['introduced'])+count(current['persistent'])
    print(json.dumps(dict(input_material_nodes_sha256=sha(x),candidate_material_nodes_sha256=sha(y),connectivity_sha256=sha(t),tags_sha256=sha(types),native_volumes_sha256=sha(ref),delta_nodes=len(ix),changed_faces=current['changed_triangles'],interface_counts=layers,material={str(tag):dict(cells=int(np.count_nonzero(types==tag)),inversions=int(np.count_nonzero(ref[types==tag]<=0)),under_minimum_volume=int(np.count_nonzero(ref[types==tag]<=1e-18)),below_quality=int(np.count_nonzero(q[types==tag]<=1e-5)),minimum_volume_m3=float(ref[types==tag].min()),minimum_scaled_determinant=float(q[types==tag].min())) for tag in (15,16)}),indent=2))
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
