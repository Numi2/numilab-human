from pathlib import Path
import json,sys
from material_context import sha
a,p,out=map(Path,sys.argv[1:]);d=json.loads((a/'manifest.json').read_text());v=json.loads((p/'manifest.json').read_text())
if sha(a/'manifest.json')!=v['source_manifest'] or 'motion_contact_registry' not in d:raise ValueError('Original factor identity mismatch')
out.mkdir()
for f in a.iterdir():
    if f.name!='manifest.json':(out/f.name).hardlink_to(f)
for n,h in v['files'].items():
    f=p/(n+'.bin');assert sha(f)==h;(out/f.name).hardlink_to(f);d['files'][f.name]=h
d['blocked_factor']=v;(out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
