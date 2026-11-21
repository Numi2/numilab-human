"""Test source skin, articular spacing and rib neighbors before fitting the composed donor skeleton."""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np
from compare_atlas import vertices_and_faces
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32
from audit_solids import winding
def run(launch,archive,out):
    cfg=json.loads(launch.read_text());m=json.loads(Path(cfg['argv'][3]).with_suffix('.manifest.json').read_text())
    archive_sha=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    assert archive_sha==m['source']['bodyparts']['archives'][0]['sha256']
    anchors=m['source']['anchors'];bank={};pins={}
    with zipfile.ZipFile(archive) as z:
        paths={Path(n).stem:n for n in z.namelist() if n.endswith('.obj')}
        def load(mid):
            buf=z.read(paths[mid]);pins[mid]=hashlib.sha256(buf).hexdigest();return vertices_and_faces(buf)
        sk,faces=load('FJ2810')
        for k,t in enumerate(anchors,1):
            if t['member_id'] not in paths:continue
            vv,ff=load(t['member_id'])
            assert pins[t['member_id']]==t['member_sha256']
            bank[k]=(vv,ff,t['member_id'])
    install_for_float32();rows=[]
    for n in (8,9,10,11):
        v,f,s=bank[n];p=intersections(v,f,sk,faces)
        x=winding(sk[faces],v[f[0]].mean(0))
        others=[]
        for k,(other,g,t) in bank.items():
            if n==k or (k in (8,9,10,11) and k<n):continue
            if np.any(v.min(0)>other.max(0)) or np.any(v.max(0)<other.min(0)):continue
            pairs=intersections(v,f,other,g)
            if pairs:others.append(dict(other=k,member=t,pairs=len(pairs),first_faces=pairs[0]))
        rows.append(dict(stable=n,member=s,skin_pairs=len(p),skin_first_faces=p[:1],winding_at_first_face=float(x),other_bone_hits=others))
    out.write_text(json.dumps(dict(source_archive_sha256=archive_sha,source_skin_sha256=pins['FJ2810'],skin_vertices=len(sk),skin_faces=len(faces),available_banks=len(bank),rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
