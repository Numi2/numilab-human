"""Bind every admitted GPU domain to its actual geometry and interval-proof input."""
import csv,json,struct,sys
from collections import Counter
from pathlib import Path
import numpy as np
from certify_heart import sha
from refine_heart import verify_parent,expected_fails

def check(work,source):
    meta=json.loads((work/'four-domains.json').read_text())
    results=json.loads((work/'refinement.json').read_text())
    assert meta['input_sha256']==sha(source)=='87bd3cfad504b0b146d14e7b4f4e5aecb03815fac55fc5c3461ec7e77ab410c7'
    v,faces=verify_parent(meta,work/'four-domains.bin')
    with source.open('rb') as f:
        nv,nf,_=struct.unpack('<3I',f.read(12));assert (nv,nf)==(len(v),len(faces))
        assert f.read(v.nbytes)==v.tobytes() and f.read(faces.nbytes)==faces.tobytes()
    wp=list(csv.DictReader((work/'four-witness.csv').open()))
    counts=np.loadtxt(work/'four-audit.log',dtype=np.int64,ndmin=2)
    assert list(counts[:,0])==list(range(len(meta['domains'])))
    assert sha(work/'four-domains.bin')==results['full_atlas_sha256']
    total=0
    for j,(parent,result) in enumerate(zip(meta['domains'],results['parents'],strict=True)):
        rows=[w for w in wp if int(w['window'])==j]
        assert j==result['parent'] and np.array_equal(counts[j,2:7],expected_fails(Counter(r['reason'] for r in rows)))
        ff=np.unique([int(w[k]) for w in rows for k in ('first_face','second_face') if int(w[k])>=0]).astype(int)
        assert np.array_equal(ff,result['witness_source_faces']) and (not rows)==result['parent_accepted_without_children']
        if not rows:continue
        vi,local=np.unique(faces[ff],return_inverse=True)
        expected_mesh=v[vi].tobytes()+local.reshape(-1,3).astype('<u4').tobytes()
        unresolved=np.array([[int(r['first_face']),int(r['second_face'])] for r in rows if r['reason']!='face'],dtype=int).reshape(-1,2)
        localpairs=np.searchsorted(ff,unresolved).astype('<u4')
        assert (work/f'parent-{j}-pairs.bin').read_bytes()==struct.pack('<I',len(localpairs))+localpairs.tobytes()
        children=result['children'];total+=len(children)
        assert not result['unproven'] and sorted(i for c in children for i in c['ids'])==sorted(parent['ids'])
        for level in sorted({c['depth'] for c in children}):
            path=work/f'parent-{j}-level-{level}'
            sums=np.loadtxt(path.with_suffix('.log'),dtype=np.int64,ndmin=2)
            with path.with_suffix('.bin').open('rb') as f:
                n,m,k=struct.unpack('<3I',f.read(12))
                assert n==len(vi) and m==len(ff) and k==len(sums)
                assert f.read(len(expected_mesh))==expected_mesh
                records=f.read()
            assert len(records)==k*116
            for c in children:
                if c['depth']!=level:continue
                i=c['record'];assert 0<=i<k and sums[i,0]==i and not sums[i,2:7].any()
                assert records[116*i:116*(i+1)]==np.asarray(c['tube'],dtype='<f4').tobytes()+struct.pack('<I',c['pivot'])
    return dict(global_faces=len(faces),global_vertices=len(v),parent_domains=len(counts),parent_pair_visits=int(counts[:,1].sum()),uncertain_witness_rows=len(wp),independently_verified_child_domains=total)
if __name__=='__main__':print(json.dumps(check(*map(Path,sys.argv[1:])),indent=2))
