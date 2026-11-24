"""Publish source-frame correction, failed shape-repair control, and the native execution boundary."""
from pathlib import Path
import sys,json,shutil
from archive_cavity_trial import archive_files,sha
def run(job,scene):
    read=lambda p:json.loads(p.read_bytes());result=read(scene/'execution.json');stage=job/'connected-back-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    scripts=['audit_current_tissue_parents.py','audit_parent_stencil_span.py','measure_shared_parent_disagreement.py','evaluate_parent_reemission.py','register_trapezius_material_frames.py','compare_connected_back_poses.py','publish_connected_back_binding.py']
    evidence=['parent-span.json','shared-parent-disagreement.json','back-pose-parity.json','tissue-pose-diagnosis/result.json','parent-reemission-trial/result.json','parent-reemission-trial/positions.npz','archived-tissue-pose-control/result.json','coupled-material-recovery.json']
    files={k:job/k for k in scripts+evidence}
    for p in scene.rglob('*'):
        if p.is_file():files['scene/'+str(p.relative_to(scene))]=p
    receipt=job/'connected-back-recovery.json'
    if not receipt.exists():archive_files(files,'connected-trapezius-source-frames-'+sha(scene/'launch.json')[:15]+'.tar.zst',receipt,'All six connected source registrations, untouched back mesh and weights, previous and updated frame controls, failed shape experiment and native launch.')
    rc=read(receipt)
    for d in [code,ev]:d.mkdir(parents=True,exist_ok=True)
    for k in scripts:shutil.copy2(job/k,code/k)
    status=(scene/'stderr.log').read_text().splitlines();stdout=(scene/'stdout.log').read_text().splitlines()
    prefix=('resting_tissue_shared_runtime=','resting_gas_transport_subcycling=','resting_vascular_external_pressure=','resting_coupled_','whole_body_skin_bone_static_bvh=','myosim_articulated_visual=','articulated_muscle_step_count=','persistent_stand_failure','registered_tissue','resting_fem_skin_bed_source=','material_cell_witness ','human_preparation_memory phase=shared_runtime_initialized')
    old=read(job/'tissue-pose-diagnosis/result.json')['surfaces'];failed=read(job/'parent-reemission-trial/result.json')['surfaces'];shared=read(job/'shared-parent-disagreement.json')
    compact=dict(device='Apple M6',launch_sha256=sha(scene/'launch.json'),execution=result,source_registration=read(scene/'frame-registration.json'),retained_configurations=read(job/'back-pose-parity.json'),previous_outside_2um=sum(v['failing'] for v in old.values()),shared_stencils_over_4um=shared['incompatible_over_4um'],maximum_split_m=shared['maximum_gap_m'],failed_direct_vertex_projection_normal_reversals=sum(v['normal_reversals'] for v in failed.values()),native_messages=[v for v in stdout+status if v.startswith(prefix)],stderr_tail=status[-12:],archive={k:rc[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'connected-trapezius-material-frame-native-result.json').write_text(json.dumps(compact,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
