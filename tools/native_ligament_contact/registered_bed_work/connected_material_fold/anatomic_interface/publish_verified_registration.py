"""Preserve the admitted material and distinguish offline registration from motion."""
import json
import shutil
from pathlib import Path
from archive_cavity_trial import archive_files, sha


def run(work):
    read = lambda p: json.loads(p.read_bytes())
    geometry = read(work / 'joint-physical-check/verification.json')
    walls = read(work / 'joint-physical-check/walls.json')
    mass = read(work / 'recorded-mass/native-replay/result.json')
    allocation = read(work / 'recorded-mass/packets/result.json')['trace']
    old = read(work.parent / 'qualified-material/bone-overlap/result.json')['results']
    if mass['exit'] or geometry['interval_violations'] or geometry['wall_crossings'] or len(old) != 185:
        raise ValueError('Native admission or registration audit failed')
    totals = {r: sum(b[r] for b in old.values()) + delta for r, delta in geometry['total_change_mm3'].items()}
    summary = dict(registered_pose=geometry, tissue_wall_triangles=walls['triangle_counts'],
                   remaining_summed_bone_overlap_mm3=totals, native_carrier_replay=allocation,
                   full_scene_qualification='Open. No accepted motion or speed measured for this candidate.',
                   upstream={s:read(work.parent/(s+'-recovery.json'))['url'] for s in ('registration','material','rebound')})
    report = work / 'verified-material.json'
    report.write_text(json.dumps(summary, indent=2)+'\n')
    receipt = work / 'recovery.json'
    keys = [report, *work.glob('*.py'), *work.glob('*.locator'), *work.glob('*.npz'), *work.glob('*.json')]
    for folder in ['current-p1','recorded-mass','selected-pose','initial-physical-check','locally-refined','refined-physical-check','joint-physical-check']:
        keys.extend(f for f in (work/folder).rglob('*') if f.is_file() and not f.is_symlink())
    if not receipt.exists():
        archive_files({str(f.relative_to(work)):f for f in keys if f!=receipt},'native-conservative-material-credits-'+sha(report)[:16]+'.tar.zst',receipt,'Exact registered-pose candidates, tissue and bone checks, native P1 material basis, source mass flows and replay. The aggressive tissue fit and rigid pelvic refit failed and are preserved for comparison.')
    base=work/'delivery'
    evidence=base/'evidence/human-connected-mesh-rebuild-20261105/m6-native-conservative-material-replay'
    source=base/'tools/native_ligament_contact/registered_bed_work/connected_material_fold/anatomic_interface'
    evidence.mkdir(parents=True,exist_ok=True)
    source.mkdir(parents=True,exist_ok=True)
    for f in [receipt,report]:shutil.copy2(f,evidence/f.name)
    for name in ['register_pelvic_sources','fit_registered_material','fit_registered_material_local','verify_material_pose','select_conservative_pose','rebuild_current_material_basis','replay_native_credits','publish_verified_registration']:
        shutil.copy2(work/(name+'.py'),source/(name+'.py'))


if __name__=='__main__':
    run(Path(__file__).resolve().parent)
