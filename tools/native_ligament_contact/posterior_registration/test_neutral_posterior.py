"""Numerical and lineage regression checks on the exact emitted muscle material carrier."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from material_context import sha
from preserve_muscle_cells import det
from guard_material_rebind import validate
from condition_motion_weights import signed_gradient
def run(candidate,fit):
    rc=json.loads((candidate/'source.json').read_text());source=Path(rc['parent']);new=candidate/source.name
    old=source.read_bytes();data=new.read_bytes();_,_,nr,nb,nv,_,_,_=struct.unpack_from('<8s6I32s',old)
    rows=np.frombuffer(old,'<u4',nr*8,64).reshape(-1,8);validate(old,data,nr,nb,nv,rows,rc)
    if sha(new)!=rc['candidate_sha256']:raise ValueError('Source content mismatch')
    rng=np.random.default_rng(76153);points=rng.normal(size=(64,3));directions=rng.normal(size=(64,3));t=np.arange(64).reshape(-1,4);ref=det(points,t);g=signed_gradient(points,t,directions,ref)
    for col in range(4):
        dx=np.zeros_like(points);dx[t[:,col]]=directions[t[:,col]];h=1.e-5
        actual=(det(points+h*dx,t)-det(points-h*dx,t))/(2*h*ref)
        np.testing.assert_allclose(g[:,col],actual,atol=1e-8,rtol=1e-8)
    def reject(where):
        b=bytearray(data);b[where]^=1
        try:validate(old,b,nr,nb,nv,rows,rc)
        except ValueError:return
        raise ValueError(('Unqualified edit admitted',where))
    bind=64+nr*32;v=bind+36*nb
    first,_,beg,_,_,_,_,_=next(r for r in rows.astype(int) if r[6]==29)
    other=next(r for r in rows.astype(int) if r[6] not in (29,30))
    for at in (64,bind+36*first,bind+36*(first+1)+4,v+56*beg,v+56*beg+40,v+56*int(other[2]),v+56*nv):reject(at)
    obs=json.loads((fit/'result.json').read_text())
    assert 0<len(obs['held_in_steps'])<len(obs['all_pose_intersections']) and obs['held_out_samples']==187
    assert all(not any(p[1:]) for p in obs['all_pose_intersections'])
    for r in obs['by_side'].values():
        assert min(r['minimum_volume_ratio'],r['minimum_normalized_determinant'])>1e-5 and r['inverted_cell_samples']==r['low_quality_cell_samples']==0
    print('Passed exact candidate lineage, seven mutation rejections, local determinant gradient and held-out pose checks')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
