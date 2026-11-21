"""Reuse certified local stiffness columns and pin the reachable nodal force domain."""
from pathlib import Path
import sys,json
from material_context import sha
def apply(base,geom,out):
    out.mkdir(exist_ok=True);before=json.loads((base/'manifest.json').read_text());d=json.loads((geom/'manifest.json').read_text())
    for key in ('material-cells.bin','material-inverse.bin','material-nodes.bin','material-modes.bin'):
        if before['files'][key]!=d['files'][key]:raise ValueError('Different material owner: '+key)
    for p in geom.iterdir():
        if p.name!='manifest.json' and not (out/p.name).exists():(out/p.name).hardlink_to(p)
    for name in ('cortical-free-index.bin','cortical-flexibility.bin'):
        f=base/name;assert sha(f)==before['files'][name];dest=out/name
        if dest.exists() and sha(dest)!=sha(f):raise ValueError('Existing distinct operator '+str(dest))
        if not dest.exists():dest.hardlink_to(f)
        d['files'][name]=sha(f)
    d['contact_flexibility']=before['contact_flexibility'];d['model']=before['model'];d['force_domain_sha256']=sha(base/'manifest.json')
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':apply(*map(Path,sys.argv[1:]))
