"""Find conservative native cardiac domains for the newly reachable pressure states."""
import argparse, hashlib, json, shutil, struct, subprocess
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from heart_domains import old_missing,tube

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_atlas(src,dest,rows):
    if dest.exists():raise FileExistsError(dest)
    subprocess.run(['cp','-c',src,dest],check=True)
    with dest.open('r+b') as f:
        n,m,_=struct.unpack('<3I',f.read(12));f.seek(8);f.write(struct.pack('<I',len(rows)));f.seek(12+n*96+m*12)
        for rec in rows:
            v,p=rec['tube'],rec['pivot'];f.write(np.array(v,'<f4').tobytes()+struct.pack('<I',p))
        f.truncate()
def main(src,forecast,table,out):
    if sha(src)!='87bd3cfad504b0b146d14e7b4f4e5aecb03815fac55fc5c3461ec7e77ab410c7':raise ValueError('Mismatched source geometry')
    raw=np.fromfile(forecast,'<f4').reshape(-1,16);q=raw[:,:7].astype(float)
    ids=old_missing(q,raw.view('<u4')[:,8],table)
    assert len(raw)==30000 and min(ids)==8200
    classes=fcluster(linkage(q[ids],method='complete',metric='chebyshev'),.040,criterion='distance')
    groups=sorted((ids[classes==c] for c in np.unique(classes)),key=np.min)
    rec=[]
    for grp in groups:
        shape,p=tube(q[grp]);rec.append(dict(ids=grp.tolist(),tube=shape.tolist(),pivot=p))
    write_atlas(src,out/'four-domains.bin',rec)
    meta=dict(input_sha256=sha(src),forecast_sha256=sha(forecast),old_table_sha256=sha(table),extended_table_input_sha256=sha(out/'four-domains.bin'),domains=rec)
    (out/'four-domains.json').write_text(json.dumps(meta,separators=(',',':'))+'\n')
    print('uncovered before',len(ids),'classes',[(len(x),int(x.min())) for x in groups],flush=True)
if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:]))
