"""Give each overlap vertex a stable material/cortex feature ID."""
import json,sys
from pathlib import Path
import numpy as np
from material_context import sha
def build(base,out):
    out.mkdir(exist_ok=True);skip={'manifest.json','pool-material-points.bin','pool-material-regions.bin'}
    for p in base.iterdir():
        if p.name not in skip and not (out/p.name).exists():(out/p.name).hardlink_to(p)
    d=json.loads((base/'manifest.json').read_text());first=d['finite_facets']['start'];n=d['finite_facets']['number']
    old=np.fromfile(base/'pool-material-points.bin','f4').reshape(-1,7,4);assert len(old)==first+2*n
    reg=np.fromfile(base/'pool-material-regions.bin','u4').reshape(-1,4);finite=reg[:,0]>=first
    assert (reg[finite,0]%2==first%2).all() and reg[finite,1].sum()==n*2
    reg[finite,0]=first+8*(reg[finite,0]-first);reg[finite,1]*=8
    raw=out/'pool-material-points.bin'
    with raw.open('wb') as f:
        old[:first].tofile(f)
        np.repeat(old[first::2],16,axis=0).tofile(f)
    rr=out/'pool-material-regions.bin';reg.tofile(rr)
    d['files'][raw.name]=sha(raw);d['files'][rr.name]=sha(rr);d['contact_pool']['count']=first+16*n
    d['finite_facets']['rows_per_pair']=16;d['finite_facets']['feature_layout']='area centroid, 3 material vertices, 3 cortical vertices, 9 opposing edge pairs'
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':build(*map(Path,sys.argv[1:]))
