"""Preserve the native shared-node contact test and complete integrated attempt."""
from pathlib import Path
import json,sys,shutil
from archive_cavity_trial import archive_files,sha
def run(job,build,scene):
    read=lambda p:json.loads(p.read_bytes())
    output=read(scene/'execution.json');test=(job/'shared-contact-test.log').read_text().strip()
    if 'GPU=Apple M6 comparisons=160 loaded_configurations=12' not in test:raise ValueError('Physical load comparison missing')
    scripts=['make_shared_tissue_contact.py','verify_shared_tissue_contact.mm','run_shared_tissue_contact.py','publish_shared_tissue_response.py']
    files={k:job/k for k in scripts+['shared-contact-test.log','initial-interval-recovery.json']}
    for k in ['app/numi-human-native','app/numilab_human_myosim_visual_probe.mm','numi_human.mm','include/numi/matter/fem_body_triangle_contact_sample.hpp','libmetalrobo.dylib','metal/NumiMatter.metallib','metal/src/metal/numi_human.metalinc','build.json','verify']:
        files['build/'+k]=build/k
    for p in scene.rglob('*'):
        if p.is_file():files['scene/'+str(p.relative_to(scene))]=p
    receipt=job/'shared-tissue-recovery.json'
    if not receipt.exists():archive_files(files,'shared-tissue-contact-'+sha(scene/'launch.json')[:15]+'.tar.zst',receipt,'Source quadrature assembly, paired native host and shaders, production GPU certificate comparison, and complete launch evidence.')
    result=read(receipt);stage=job/'shared-tissue-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in [code,ev]:d.mkdir(parents=True,exist_ok=True)
    for k in scripts:shutil.copy2(job/k,code/k)
    logs=(scene/'stdout.log').read_text()+'\n'+(scene/'stderr.log').read_text()
    markers=('registered_cortical_contact=','complete_bone_bvh','myosim_articulated_visual=','articulated_muscle_step_count=','whole_body_skin_fat_registered_volume_install=','registered_loaded_material_install=','whole_body_skin_bone_static','resting_fem_skin_bed_source=')
    summary=dict(device='Apple M6 / SSH Numi',parent=read(scene/'launch.json')['parent_launch'],launch=sha(scene/'launch.json'),build=sha(build/'build.json'),execution=output,gpu_contact_parity=test,law='On the same slave coordinate and closed master BVH, sum disjoint source areas, integrate material stiffness against area, and retain the largest constituent pressure and strain slopes. Closure counts measure unique active nodal queries.',source_volume_overlap=read(job/'initial-interval-delivery/evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123/native-initial-material-interval-and-organs.json')['bone_tissue_overlap'],native_witnesses=[v for v in logs.splitlines() if v.startswith(markers)],native_log_tail=logs.splitlines()[-7:],archive={k:result[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'shared-tissue-contact-and-native-scene.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
