"""Retain the complete, independently replayed and natively admitted source-to-tissue mass transfer."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha
def run(root):
    read=lambda s:json.loads((root/s).read_bytes());scripts=['audit_material_edge_support.py','exchange_material_credits.py','prepare_counted_mass_allocator.py','replay_connected_allocation.py','run_material_support.py','publish_admitted_material_mass.py']
    files={s:root/s for s in scripts+['counted-mass-allocator.cpp','counted-mass-allocator','counted-mass-allocator.log','registered-quadrature-recovery.json','native-coupled-allocation/stderr.log']}
    for d in ['feasible-carriers','feasible-quadrature','native-unique-allocation']:
        for f in (root/d).rglob('*'):
            if f.is_file():files[str(f.relative_to(root))]=f
    for s in ['launch.json','stdout.log','stderr.log','execution.json']:files['scene-live-tissue-mass/'+s]=root/'scene-live-tissue-mass'/s
    archive=root/'accepted-material-mass-recovery.json'
    if not archive.exists():archive_files(files,'native-connected-tissue-mass-'+sha(root/'scene-live-tissue-mass/launch.json')[:16]+'.tar.zst',archive,'Verified connected skin/fat material source routes, native mass admission, and subsequent rejection of the coarse physical parent reference.')
    rec=read(archive.name);flow=read('feasible-carriers/result.json');native=read('native-unique-allocation/result.json')
    out=(root/'scene-live-tissue-mass/stdout.log').read_text().splitlines();err=(root/'scene-live-tissue-mass/stderr.log').read_text().splitlines()
    result=dict(device='SSH Numi Apple M6',native_verifier=native['receipts'],independent_replay=(root/'native-unique-allocation/stdout.log').read_text().splitlines(),whole_scene_mass=[s for s in out if s.startswith(('registered_volume_current_fractional_recorded_carrier_allocation=','registered_volume_fractional_mass='))],reconciled_routes=flow,blocking_guard=next(s for s in err if s.startswith('myosim_articulated_visual=failed')),native_steps=0,recovery={k:rec[k] for k in ('url','archive_sha256','size_bytes')},qualification='Current two-role material allocation passes the independent native writer/readers and full-scene receiving code. The coarse parent consistency check still blocks replacement before stepping.')
    out=root/'admitted-material-mass-delivery';code=out/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=out/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123';code.mkdir(parents=True,exist_ok=True);ev.mkdir(parents=True,exist_ok=True)
    for s in scripts:shutil.copy2(root/s,code/s)
    (ev/'native-connected-skin-fat-mass-admission.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
