"""Publish native material admission and exactly the execution boundary reached with this loaded body."""
from pathlib import Path
import sys,json,shutil
from archive_cavity_trial import archive_files,sha
def run(job,folder):
    read=lambda p:json.loads(p.read_bytes());out=folder/'scene';build=folder/'native';result=read(out/'execution.json')
    scripts=['verify_registered_organ_identity.py','bind_registered_organ_records.py','certify_initial_fat_interval.py','build_repaired_native_pose.py','integrate_certified_fat_pose.py','publish_initial_material_certificate.py']
    files={k:job/k for k in scripts+['deep-fat-material-scene-recovery.json','prostate-registration.json','registered-organ-viewer/records.json','registered-organ-viewer/build.json','registered-organ-viewer/app/numilab_human_myosim_visual_probe.mm','registered-organ-scene/launch.json','registered-organ-scene/stdout.log','registered-organ-scene/stderr.log','registered-organ-scene/execution.json','executable-fat-pose/current.node','executable-fat-pose/initial-layers.json','executable-fat-pose/result.json']}
    for sub in ['contact','cortex','bone-audit','scene']:
        for f in (folder/sub).rglob('*'):
            if f.is_file():files[str(f.relative_to(job))]=f
    for f in [build/'app/NumiHumanLoadedSkinPoseData.inc',build/'app/NumiHumanCorticalSkinBinding.inc',build/'app/numi-human-native',build/'build.json',folder/'material/result.json']:files[str(f.relative_to(job))]=f
    receipt=job/'initial-interval-recovery.json'
    if not receipt.exists():archive_files(files,'certified-physical-shoulder-'+sha(out/'launch.json')[:15]+'.tar.zst',receipt,'Original organ equivalence, coupled material first-pose interval correction, recovered native binary, exact launch and physical run evidence.')
    rc=read(receipt);stage=job/'initial-interval-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in (code,ev):d.mkdir(parents=True,exist_ok=True)
    for s in scripts:shutil.copy2(job/s,code/s)
    stdout=(out/'stdout.log').read_text();stderr=(out/'stderr.log').read_text()
    bone=read(folder/'bone-audit/result.json')
    evidence=dict(device='Apple M6 on SSH Numi',launch_sha256=sha(out/'launch.json'),binary_sha256=sha(build/'app/numi-human-native'),execution=result,source_records=read(job/'registered-organ-viewer/records.json'),starting_pose=read(job/'executable-fat-pose/result.json'),bone_tissue_overlap={k:dict(positive_bones=bone['summary'][k]['positive_banks'],unresolved=bone['summary'][k]['unresolved'],sum_pair_volume_cm3=sum(r[k] for r in bone['results'].values())/1000) for k in ['skin','fat']},native_messages=[s for s in (stdout+'\n'+stderr).splitlines() if s.startswith(('resting_prostate_fem=','resting_passive_organ_batch=','resting_optic_nerve_fem=','registered_loaded_material_install=','whole_body_skin_fat_registered_volume_install=','resting_fem_skin_bed_source=','myosim_articulated_visual=','articulated_muscle_step_count=','material_cell_witness '))],recovery={k:rc[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'native-initial-material-interval-and-organs.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
