"""Retain the complete registered material launch and the result of its actual native execution."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(job,scene,build):
    rd=lambda f:json.loads(f.read_bytes());execution=rd(scene/'execution.json')
    scripts=['register_loaded_cortical_interfaces.py','compile_cortical_skin_contact.py','build_cortical_skin_registration.py','measure_bone_tissue_overlap.py','register_repaired_fat_nodes.py','reproject_repaired_fat_support.py','prepare_repaired_tissue_ledger.py','build_repaired_native_pose.py','run_repaired_fat_scene.py','publish_registered_deep_fat_scene.py']
    files={s:job/s for s in scripts+['deep-fat-recovery.json','registered-frame-recovery.json','deep-fat-loaded-bone-overlap/result.json','deep-fat-bone-overlap/result.json']}
    for name in ['deep-fat-p1','deep-fat-registered-tables','deep-fat-cortex','deep-fat-cortex-contact','deep-fat-ledger/packets','deep-fat-ledger/flows','deep-fat-ledger/native-replay']:
        for f in (job/name).rglob('*'):
            if f.is_file():files[str(f.relative_to(job))]=f
    for path in [build/'build.json',build/'app/NumiHumanCorticalSkinBinding.inc',build/'app/NumiHumanLoadedSkinPoseData.inc',build/'app/numi-human-native']:files[str(path.relative_to(job))]=path
    for base in [job/'deep-fat-scene',job/'deep-fat-contact-scene',scene]:
        for fn in ['launch.json','stdout.log','stderr.log','execution.json','registered-initial-bodies.json']:
            f=base/fn
            if f.exists():files[str(f.relative_to(job))]=f
    for f in (scene/'native-traces').rglob('*'):
        if f.is_file():files[str(f.relative_to(job))]=f
    receipt=job/(scene.name+'-recovery.json')
    if not receipt.exists():archive_files(files,'registered-deep-fat-scene-'+sha(scene/'launch.json')[:15]+'.tar.zst',receipt,'Native integrated skin and fat, registered pose and contact, conserved mass replay and actual runtime evidence.')
    rc=rd(receipt);code=job/(scene.name+'-delivery')/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=job/(scene.name+'-delivery')/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in (code,ev):d.mkdir(parents=True,exist_ok=True)
    for s in scripts:shutil.copy2(job/s,code/s)
    logs=(scene/'stdout.log').read_text().splitlines();errors=(scene/'stderr.log').read_text().splitlines()
    bf=rd(job/'deep-fat-loaded-bone-overlap/result.json');over={k:dict(bones=bf['summary'][k]['positive_banks'],indeterminate=bf['summary'][k]['unresolved'],sum_pair_overlap_cm3=sum(v[k] for v in bf['results'].values())/1000) for k in ('skin','fat')}
    summary=dict(device='SSH Numi Apple M6',execution=execution,launch_sha256=sha(scene/'launch.json'),binary_sha256=sha(build/'app/numi-human-native'),native_mass_receipts=rd(job/'deep-fat-ledger/native-replay/result.json')['receipts'],p1_registration=rd(job/'deep-fat-p1/result.json')['receipts'],initial_loaded_bone_volume_overlap=over,source_layers=rd(job/'deep-fat-native/result.json')['layer_intersections'],native_scene_results=[s for s in logs if s.startswith(('registered_volume_native_preflight=','registered_volume_fractional_mass=','registered_volume_source_identity_binding=','registered_loaded_material_install=','registered_volume_inherited_D_constraint_rows=','whole_body_skin_fat_registered_volume_install=','myosim_articulated_visual=','articulated_muscle_step_count='))],termination=[s for s in errors if s.startswith(('myosim_articulated_visual=failed','registered_volume_native_preflight=FAIL'))],recovery={k:rc[k] for k in ('url','archive_sha256','size_bytes')})
    (ev/(scene.name+'-result.json')).write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
