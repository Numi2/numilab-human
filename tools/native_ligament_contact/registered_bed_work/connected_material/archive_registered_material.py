"""Recover the registered-pose experiment and verify all archived bytes by independent download."""
import json
from pathlib import Path
from archive_cavity_trial import archive_files,sha

root=Path(__file__).resolve().parent
files={}
def add(p,key=None):
    if not p.is_file():raise FileNotFoundError(p)
    files[key or str(p.relative_to(root))]=p

for p in sorted(root.iterdir()):
    if p.is_file() and (p.suffix in ('.py','.cpp','.hpp','.dylib') or p.name in
        ('run-tissue-registration','decision.json','retired-comparison-space.json','retired-comparison-locations.json','closed-frames.json','closed-recovery.json')):
        add(p)
for f in ('matched-pose/launch.json','matched-pose/pose-choice.json','installed-assigned/assignment.json','installed-assigned/registered-stock.csv',
          'assigned-source/launch.json','calibrated/recruited-source/launch.json','calibrated/opening-muscle-load.json','calibrated/foundation.json',
          'measured-opening-source/witness.json.opening.csv','pose-one/registered-bed-columns.csv','pose-one/resting-local-bed-footprint.csv',
          'pose-one/resting-local-bed-sections.csv','registered-79/registered-bed-columns.csv',
          'registered-79/initial.mrvpack.zst','registered-79/accepted.mrvpack.zst'):
    add(root/f)
for name in ('prior-pose-paths','native-pose-paths','installed-prior','installed-matched'):
    for f in ('connected-columns.json','paths.npz'):add(root/name/f)
for n in ('registered-79','pose-one'):
    for f in ('skin-donors.json','skin-donors.npz','envelope/accepted-envelope.json','envelope/physical-banks.json','envelope/target-witnesses.jsonl','resting-ocular-sclera-window.json'):
        add(root/n/f)
add(root/'registered-79/shoulder-terms.npz')
add(root/'registered-79/shoulder-motion.json')
for name in ('pose-owner','stock-owner','calibrated/observer','calibrated/native'):
    d=root/name
    identities=json.loads((d/'identities.json').read_text())
    for f,h in identities.items():
        if sha(d/f)!=h:raise ValueError('Stale '+str(d/f))
        add(d/f)
    for f in ('identities.json','build-commands.json','shader-build-commands.json'):add(d/f)
for name in ('pose-footprint','pose-one','pose-control-16','stock-control-16','measured-opening','recruited-16','registered-79','launcher-79'):
    for f in ('launch.json','execution.json','terminal.receipt.json','terminal-bed-certificate.json','failure.json','resting-coupled.csv',
              'resting-surface-audit.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv','stdout.log','stderr.log'):
        p=root/name/f
        if p.is_file():add(p)
    for f in ('launch.json','witness.json','opening-bodies.json'):
        p=root/(name+'-source')/f
        if p.is_file():add(p)
for p in (root.parent/'recovery.json',root.parents[1]/'endpoint-recovery.json',root.parents[2]/'finite-recovery.json'):
    add(p,'parent/'+p.name)
archive_files(files,'connected-material-'+sha(root/'decision.json')[:16]+'.tar.zst',root/'recovery.json',
              'Native snapshots, source, compiled programs, verified installed connectivity, complete trials and previous recovery links. Native motion uses the reduced skin/fat support owner.')
