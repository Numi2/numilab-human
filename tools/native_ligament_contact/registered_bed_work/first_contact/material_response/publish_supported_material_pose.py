"""Preserve a reproducible material pose and both failed alternatives."""
import json, shutil, sys
from pathlib import Path
from archive_cavity_trial import archive_files, sha

def run(root):
    def read(name):
        return json.loads((root/name).read_bytes())
    scripts=['fit_distal_arm_posture.py','fit_material_pose.py','certify_pose_lattice.py','certify_transported_material.py','install_supported_material_pose.py','capture_supported_posture.py','compare_supported_postures.py','measure_bone_tissue_overlap.py','register_limb_field.py','publish_supported_material_pose.py','fit_supported_arm_posture.py','trial_supported_arm_posture.py']
    archives=scripts+['distal-registered/result.json','distal-registered-native/launch.json','distal-registered-native/target-pose.json','bounded-final/result.json','bounded-final/material-transport.npz','physical-pose/result.json','physical-pose/material-transport.npz','whole-overlap/result.json','registered-overlap/result.json','fitted-overlap/result.json','precision/result.json','joint-comparison.json','distal-comparison.json','recovery.json','posed-app/current.node','posed-app/result.json','posed-app/app/NumiHumanLoadedSkinPoseData.inc','posed-app/app/NumiHumanCorticalRegistration.inc','posed-app/app/numi-human-native','posed-app/scene/launch.json','posed-app/scene/execution.json','posed-app/scene/stderr.log','posed-app/scene/registered-initial-bodies.json']
    receipt=root/'pose-recovery.json'
    if not receipt.exists():
        archive_files({s:root/s for s in archives},'supported-material-pose-'+sha(root/'posed-app/result.json')[:16]+'.tar.zst',receipt,'Complete connected skin/fat resting-pose field, physical reference and mass identities, staged native binary and launch, matched cortical overlap comparisons and failed alternatives.')
    rec=read(receipt.name)
    if not rec['verified_member_sha256']:raise ValueError('Recovery verification incomplete')
    dest=root/'pose-delivery';code=dest/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=dest/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    code.mkdir(parents=True,exist_ok=True);ev.mkdir(parents=True,exist_ok=True)
    for f in scripts:shutil.copy2(root/f,code/f)
    completed=read('posed-app/result.json');completed.pop('commands')
    if any(completed['shape'].values()) or any(x['crossings'] for x in completed['curves']):raise ValueError('Physical pose failed')
    log=(root/'posed-app/scene/stderr.log').read_text().splitlines()
    summary={k:read(''+k+'/result.json')['summary'] for k in ['whole-overlap','fitted-overlap','registered-overlap']}
    record=dict(device='SSH Numi Apple M6',material_pose=completed,lattice=read('physical-pose/result.json'),pose=read('distal-registered/result.json'),overlap_mm3=summary,prior=read('joint-comparison.json'),actual=read('distal-comparison.json'),ligament_admission=next(x for x in log if x.startswith('registered_ligament_source_geometry=')),scene_admission=next(x for x in log if x.startswith('registered_volume_native_preflight=FAIL')),steps=0,qualification='Native skeletal pose and static volumetric geometry. Skin and fat material at rest retain their authored masses. Full material import, equilibrium, motion and sustained performance remain open.',recovery={k:rec[k] for k in ('url','archive_sha256','size_bytes')})
    (ev/'supported-material-pose-and-overlap.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
