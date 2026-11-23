"""Archive new material-frame comparisons, native traction experiments and exact source landmarks."""
import json
from pathlib import Path
import sys
from archive_cavity_trial import archive_files,sha

root=Path(__file__).resolve().parent
control,later=map(lambda x:Path(x).resolve(),sys.argv[1:])
files={}
def add(p,k=None):
    if not p.is_file():raise FileNotFoundError(p)
    files[k or str(p.relative_to(root))]=p
for name in ('archive_shear_qualification.py','build_shear_observer.py','opening_shear.py','compile_load_recruitment.py','prepare_connected_columns.py',
             'certify_registered_release.py','certify_shear.py','rank_opening_owners.py','audit_material_boundaries.py','audit_route_clearance.py','test_route_clearance.py',
             'compare_native_route_skin.py','verify_bone_enclosure.py','decision.json','recovery.json','shear/decision.json'):
    add(root/name)
for p in (root/'material-source').rglob('*'):
    if p.is_file() and p.suffix in ('.json','.jsonl','.npz'):add(p)
for case,name in ((control,'dense-control'),(later,'dense-later')):
    for f in ('launch.json','native-initial-bodies.json','registered-initial-bodies.json','execution.json'):add(case/f,name+'/'+f)
add(later.parent/'dense-recovery.json','dense-later/parent-recovery.json')
add(later.parent/'dense-result.json','dense-later/control-and-trial.json')
for n in ('shear/observer','shear/loaded/native','shear/refined/native'):
    p=root/n;ids=json.loads((p/'identities.json').read_text())
    for f,h in ids.items():
        if sha(p/f)!=h:raise ValueError('Stale native '+n)
        add(p/f)
    for f in ('identities.json','build-commands.json','shader-build-commands.json'):add(p/f)
for case in ('shear/control-one','shear/loaded/one','shear/loaded/fullaudit','shear/loaded/matched-79','shear/refined/one','shear/refined/run-16','shear/refined/accepted-82','shear/refined/matched-79'):
    p=root/case
    for f in ('launch.json','execution.json','failure.json','terminal.receipt.json','terminal-bed-certificate.json','stdout.log','stderr.log','resting-coupled.csv','resting-com-support-impulses.csv','resting-com-momentum-diagnostic.csv','resting-surface-audit.csv','shoulder-motion.json','shoulder-terms.npz','accepted.mrvpack.zst','resting-ocular-sclera-window.json','envelope/accepted-envelope.json','envelope/physical-banks.json','envelope/target-witnesses.jsonl'):
        if (p/f).is_file():add(p/f)
    src=p.with_name(p.name+'-source')
    for f in ('launch.json','witness.json','witness.json.opening.csv'):
        if (src/f).is_file():add(src/f)
for n in ('shear/loaded','shear/refined'):
    for f in ('source/launch.json','opening-muscle-load.json','foundation.json'):add(root/n/f)
archive_files(files,'registered-traction-'+sha(root/'shear/decision.json')[:16]+'.tar.zst',root/'shear/recovery.json',
              'Registered pose and loaded-phase skin, 878-bank opening comparisons, clinical-free reference route locations, two optional initialization experiments. The control archive supplies shared source assets and initial geometry.')
