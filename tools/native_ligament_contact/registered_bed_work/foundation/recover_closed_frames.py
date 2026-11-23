"""Retire closed old viewer exports only after independent GitHub restoration."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha
root,case,out=(Path(x).resolve() for x in sys.argv[1:]);free=shutil.disk_usage(root).free
jobs=[root/'resting-clearance-and-atrium-016'/n/'native/resting-human.mrvpack' for n in ('motion-backtracking','motion-slot-410','motion-contact-witness')]
protected=set()
for path in (case/'launch.json',out.parent/'live-source/launch.json',out.parents[1]/'dense-four/launch.json'):
    cfg=json.loads(path.read_text());s=[*cfg.get('pins',{}),*cfg['argv'],*cfg['environment'].values()]
    protected.update(str(Path(k).resolve()) for k in s if isinstance(k,str) and k.startswith('/'))
manifest={};members={};count=0
for p in jobs:
    if not p.is_file() or p.is_symlink() or str(p.resolve()) in protected:raise ValueError('historical capture is missing or in use')
    h=sha(p);manifest[str(p.relative_to(root))]=h
    members['geometry/'+h+'.mrvpack']=p;count+=p.stat().st_size
m=out/'closed-packed-locations.json';m.write_text(json.dumps(manifest,indent=2)+'\n');members['locations.json']=m
archive_files(members,'closed-fem-opening-'+sha(m)[:16]+'.tar.zst',out/'closed-packed-recovery.json','Exact original bytes of superseded opening packs from three closed case 016 trials. Local paths are in locations.json. All sources, accepted captures and active model inputs are preserved.')
proof=json.loads((out/'closed-packed-recovery.json').read_text())
if proof['verified_member_sha256']['locations.json']!=sha(m):raise ValueError('unrecoverable locations')
for n,h in manifest.items():
    if proof['verified_member_sha256']['geometry/'+h+'.mrvpack']!=h or sha(root/n)!=h:raise ValueError('failed off-machine recovery')
for n in manifest:(root/n).unlink()
(out/'closed-packed-space.json').write_text(json.dumps(dict(files=len(manifest),payloads=len(members)-1,logical_bytes=count,free_before_bytes=free,free_after_bytes=shutil.disk_usage(root).free),indent=2)+'\n')
