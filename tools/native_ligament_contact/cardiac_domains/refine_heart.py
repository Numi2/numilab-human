"""Certify residual face witnesses on nested cardiac domains.

The native gate intersects a parent domain with each child. The full parent
proof covers all faces and pairs except its complete list of witnesses.
Every unresolved face and both members of every unresolved pair are kept in
the child mesh. The affine modes and shared vertex identities stay exact.
"""
import csv,hashlib,json,struct,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from heart_domains import tube
from certify_heart import sha

def write_subset(dst,vertices,faces,domains):
    with dst.open('wb') as f:
        f.write(struct.pack('<3I',len(vertices),len(faces),len(domains)))
        f.write(vertices.tobytes());f.write(faces.tobytes())
        for d,p,ids in domains:f.write(d.astype('<f4').tobytes()+struct.pack('<I',p))
def split(ids,q):
    if len(ids)<=1:return []
    z=linkage(q[ids],method='complete',metric='chebyshev')
    # Cut the top linkage edge, including ties without empty sets.
    from scipy.cluster.hierarchy import to_tree
    root=to_tree(z)
    return [ids[np.array(child.pre_order(),int)] for child in (root.left,root.right)]
def expected_fails(rows):
    return [rows['face'],*[rows[str(k)] for k in (1,2,3,4)]]
def verify_parent(meta,atlas):
    assert sha(atlas)==meta['extended_table_input_sha256']
    with atlas.open('rb') as f:
        n,m,count=struct.unpack('<3I',f.read(12))
        v=np.fromfile(f,'<f4',n*24).reshape(n,8,3);faces=np.fromfile(f,'<u4',m*3).reshape(m,3)
        assert count==len(meta['domains'])
        for rec in meta['domains']:
            assert np.array_equal(np.fromfile(f,'<f4',28).reshape(4,7),rec['tube']) and struct.unpack('<I',f.read(4))[0]==rec['pivot']
        assert not f.read(1)
    return v,faces

def main(work,probe):
    meta=json.loads((work/'four-domains.json').read_text())
    assert sha(work/'coordinates.bin')==meta['forecast_sha256']
    v,faces=verify_parent(meta,work/'four-domains.bin')
    q=np.fromfile(work/'coordinates.bin','<f4').reshape(-1,16)[:,:7].astype(float)
    with (work/'four-witness.csv').open() as f:wr=list(csv.DictReader(f))
    lines=[list(map(int,l.split())) for l in (work/'four-audit.log').read_text().splitlines()]
    assert [r[0] for r in lines]==list(range(len(meta['domains'])))
    receipts=[]
    for parent,row in zip(meta['domains'],lines,strict=True):
        j=row[0];w=[r for r in wr if int(r['window'])==j]
        assert expected_fails(Counter(r['reason'] for r in w))==row[2:7], 'Parent witness truncation'
        assert all(len(r)==9 for r in [row]) and all(i>=0 for i in row)
        ids=np.array(parent['ids']);children=[];unproven=[]
        face_ids=np.unique([int(r[k]) for r in w for k in ('first_face','second_face') if int(r[k])>=0])
        if len(face_ids):
            assert len(faces)>face_ids[-1]
            vertices,compact=np.unique(faces[face_ids],return_inverse=True)
            local=v[vertices];localfaces=compact.reshape(-1,3).astype('<u4')
            assert np.array_equal(vertices[localfaces],faces[face_ids])
            pairfile=work/f'parent-{j}-pairs.bin'
            selected=np.array([(int(r['first_face']),int(r['second_face'])) for r in w if r['reason']!='face'],dtype=np.int32).reshape(-1,2)
            compactpairs=np.searchsorted(face_ids,selected).astype('<u4')
            assert np.array_equal(face_ids[compactpairs],selected)
            with pairfile.open('wb') as f:f.write(struct.pack('<I',len(selected)));f.write(compactpairs.tobytes())
            # Recheck every face in this submesh and every pair left undecided by the full parent.
            initial=fcluster(linkage(q[ids],method='complete',metric='chebyshev'),.009,criterion='distance') if len(ids)>1 else [1]
            pending=[ids[np.asarray(initial)==c] for c in np.unique(initial)]
            depth=0
            while pending:
                candidates=[(*tube(q[g]),g) for g in pending]
                stem=work/f'parent-{j}-level-{depth}'
                atlas=stem.with_suffix('.bin');out=stem.with_suffix('.csv');log=stem.with_suffix('.log')
                write_subset(atlas,local,localfaces,candidates)
                with log.open('w') as h:subprocess.run([probe,atlas,out,'split-tube',pairfile],stdout=h,check=True)
                scores=[list(map(int,s.split())) for s in log.read_text().splitlines()]
                assert len(scores)==len(candidates)
                next_groups=[]
                for k,((d,p,pts),s) in enumerate(zip(candidates,scores,strict=True)):
                    assert s[0]==k and len(s)==9
                    if not any(s[2:7]):
                        children.append(dict(tube=d.tolist(),pivot=p,ids=pts.tolist(),depth=depth,record=k))
                    elif len(pts)>1:next_groups.extend(split(pts,q))
                    else:unproven.extend(pts.tolist())
                pending=next_groups;depth+=1
        success=sum((c['ids'] for c in children),[]) if w else ids.tolist()
        assert sorted(success+unproven)==sorted(ids)
        rec=dict(parent=j,witness_faces=len(face_ids),witness_rows=len(w),witness_source_faces=face_ids.tolist(),parent_accepted_without_children=not w,children=children,unproven=unproven)
        receipts.append(rec)
        print('parent',j,'source faces',len(face_ids),'admitted states',len(success),'children',len(children),'unresolved',len(unproven),flush=True)
    (work/'refinement.json').write_text(json.dumps(dict(full_atlas_sha256=meta['extended_table_input_sha256'],auditor_sha256=sha(probe),parents=receipts),separators=(',',':'))+'\n')
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
