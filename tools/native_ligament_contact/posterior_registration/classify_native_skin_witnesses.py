"""Classify existing exact skin witnesses on the same emitted Float32 triangles."""
import json,mmap,runpy,sys
from collections import Counter
from pathlib import Path
import numpy as np
from count_accepted_cortical_crossings import dyadic,plane,straddles
from material_context import sha

def classify(impl,pack,report,out):
    sys.path.insert(0,str(impl.parent/"src"))
    audit=runpy.run_path(str(impl));src=json.loads(report.read_text())
    receipt=json.loads(pack.with_suffix(".receipt.json").read_text())
    if receipt["pack_file_sha256"]!=src["accepted_pack_sha256"] or sha(pack)!=src["accepted_pack_sha256"]:raise ValueError('Mismatched skin capture')
    poses={v["body_index"]:v for v in receipt["accepted_body_poses"]}
    def count(v,f,w,g,row):
        hits=row["triangle_pairs"];transverse=[]
        if len(hits)!=row["count"]:raise ValueError('Expected all intersecting triangle pairs')
        for i,j in hits:
            xyz=dyadic(np.concatenate((v[f[i]],w[g[j]])))
            a=xyz[:3];b=xyz[3:]
            if straddles(a,b[0],plane(*b)) and straddles(b,a[0],plane(*a)):transverse.append([i,j])
        return {"intersecting":len(hits),"transverse":len(transverse),"boundary_only":len(hits)-len(transverse),"first_transverse":transverse[:3]}
    rows={(r['semantic'],r['stable_id'],r['primitive']):r for r in src['surfaces']}
    result={"step":src['accepted_step'],"geometry_sha256":src['accepted_pack_sha256'],"banks":[]}
    with pack.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
        skin=list(audit['primitives'](m,poses,skin_only=True));assert len(skin)==1
        _,_,x,faces=skin[0];result['self']=count(x,faces,x,faces,src['self_intersections'])
        total=Counter()
        for (semantic,stable),primitive,v,g in audit['primitives'](m,poses):
            if (semantic,stable)==audit['SKIN']:continue
            target=rows.pop((semantic,stable,primitive));r=count(x,faces,v,g,target)
            if not target['ocular_source_id']:
                for k in ['intersecting','transverse','boundary_only']:total[k]+=r[k]
            if r['intersecting']:result['banks'].append(dict(semantic=semantic,stable_id=stable,primitive=primitive,**r))
    if rows:raise ValueError("Missing native primitives")
    result['nonocular']=dict(total)
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':classify(*map(Path,sys.argv[1:]))
