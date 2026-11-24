"""Preserve verified respiratory and cardiac registrations and the resulting native execution."""
from pathlib import Path
import sys,json,shutil
from archive_cavity_trial import archive_files,sha
def run(job,base,scene):
    read=lambda p:json.loads(p.read_bytes());result=read(scene/'execution.json');reg=read(base/'intercostal-registration.json');heart=read(job/'current-cardiac-records.json')
    scripts=['register_current_intercostal_attachments.py','compare_current_cardiac_records.py','install_current_cardiac_field.py','publish_coupled_material_binding.py']
    files={k:job/k for k in scripts+['current-cardiac-records.json','current-solver-binding-recovery.json']}
    for k in ('intercostal-registration.json','launch.json','execution.json','stderr.log','stdout.log'):files['rib/'+k]=base/k
    for p in scene.rglob('*'):
        if p.is_file():files['scene/'+str(p.relative_to(scene))]=p
    receipt=job/'coupled-material-recovery.json'
    if not receipt.exists():archive_files(files,'registered-heart-and-ribs-'+sha(scene/'launch.json')[:15]+'.tar.zst',receipt,'Native verified respiratory and cardiac relations, immutable source and physics bindings, heart maps, full scene launch and original trial outcome.')
    recovered=read(receipt);stage=job/'coupled-material-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in [code,ev]:d.mkdir(parents=True,exist_ok=True)
    for k in scripts:shutil.copy2(job/k,code/k)
    so=(scene/'stdout.log').read_text();se=(scene/'stderr.log').read_text();allLines=(so+'\n'+se).splitlines()
    prefixes=('registered_cortical_contact=','whole_body_skin_bone_static_bvh=','resting_tissue_shared_runtime=','myosim_articulated_visual=','articulated_muscle_step_count=','resting_atrial','common_cardiac_','resting_gas_transport_subcycling=','resting_fem_skin_bed_source=','resting_physiology','material_cell_witness ','resting_coupled_','human_preparation_memory phase=shared_runtime_initialized')
    summary=dict(launch=sha(scene/'launch.json'),parent=sha(base/'launch.json'),trial=result,rib_attachments=reg,cardiac_ranges=len(heart['ranges']),cardiac_volume_and_wall_ranges=sum(r['physical'] for r in heart['ranges']),cardiac_vertices_compared=sum(v['vertices'] for v in heart['ranges']),cardiac_changed_vertices=sum(v['changed_vertices'] for v in heart['ranges']),changed_triangle_sets=sum(not v['triangle_parity'] for v in heart['ranges']),terminal_messages=[v for v in allLines if v.startswith(prefixes)],stderr_tail=se.splitlines()[-8:],archive={k:recovered[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'registered-heart-ribs-and-native-execution.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
