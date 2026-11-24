"""Relink the admitted pose executable against the already compiled read-only residual observer."""
from pathlib import Path
import json,subprocess,sys
from build_bed_audit import sha
def run(current,base,observer,out):
    pins=json.loads((observer/'pins.json').read_bytes())
    if any(sha(Path(p))!=h for p,h in pins.items()):raise ValueError('Observer source mismatch')
    d=json.loads((current/'build.json').read_text());cmd=d['commands'][-1];app=out/'app';app.mkdir(parents=True,exist_ok=True)
    (out/'runtime').symlink_to(observer/'runtime');(out/'rows.metallib').symlink_to(observer/'rows.metallib')
    swaps={str(base/'app/app-runtime.mm.o'):str(observer/'app/app-runtime.mm.o'),str(base/'runtime/libmetalrobo.dylib'):str(observer/'runtime/libmetalrobo.dylib'),str(current/'numi-human-native'):str(app/'numi-human-native')}
    if any(cmd.count(k)!=1 for k in swaps):raise ValueError('Different physics link')
    cmd=[swaps.get(s,s) for s in cmd];subprocess.run(cmd,check=True)
    files=(app/'numi-human-native',out/'rows.metallib',out/'runtime/libmetalrobo.dylib')
    (out/'pins.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n')
    (out/'link.json').write_text(json.dumps(cmd,indent=2)+'\n')
if __name__=='__main__':run(*(Path(p).resolve()for p in sys.argv[1:]))
