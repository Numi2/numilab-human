"""Publish executable tissue ownership and the exact integrated outcome on Numi."""
from pathlib import Path
import json,sys,shutil
from archive_cavity_trial import archive_files,sha
def run(job,build,scene,domain):
    read=lambda p:json.loads(p.read_bytes());info=read(build/'build.json');actual=read(scene/'execution.json')
    files={k:job/k for k in ['bind_certified_volume_owner.py','measure_registered_contact_domain.py','publish_current_solver_binding.py','shared-tissue-recovery.json']}
    for k in ['runtime.mm','app-runtime.mm','bed_response.inc','app/numi-human-native','libmetalrobo.dylib','build.json']:
        files['build/'+k]=build/k
    files['contact-domain.json']=domain
    for p in scene.rglob('*'):
        if p.is_file():files['scene/'+str(p.relative_to(scene))]=p
    archive=job/'current-solver-binding-recovery.json'
    if not archive.exists():archive_files(files,'current-registered-volume-solver-'+sha(scene/'launch.json')[:15]+'.tar.zst',archive,'Complete checked connectivity and material binding, preserved native bounds, contact geometry measurement, executable and run artifacts.')
    receipt=read(archive);stage=job/'current-solver-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for folder in [code,ev]:folder.mkdir(parents=True,exist_ok=True)
    for k in ['bind_certified_volume_owner.py','measure_registered_contact_domain.py','publish_current_solver_binding.py']:shutil.copy2(job/k,code/k)
    kw=('myosim_articulated_visual=','registered_cortical_contact=','registered_loaded_material_install=','whole_body_skin_fat_registered_volume_install=','whole_body_skin_bone_static_bvh=','resting_shared_skin_fat_preconditioner=','articulated_muscle_step_count=','human_preparation_memory phase=','material_cell_witness ','resting_fem_skin_bed_source=')
    logs=(scene/'stdout.log').read_text()+'\n'+(scene/'stderr.log').read_text()
    summary=dict(runtime={k:v for k,v in info.items() if k!='commands'},launch=sha(scene/'launch.json'),execution=actual,contact_work=read(domain),native_witnesses=[line for line in logs.splitlines() if line.startswith(kw)],stderr_tail=(scene/'stderr.log').read_text().splitlines()[-12:],archive={k:receipt[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'registered-physical-volume-solver-result.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
