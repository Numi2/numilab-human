"""Record actual native execution with the certified anatomy and pressure domain."""
from pathlib import Path
import sys,json,shutil,csv
from archive_cavity_trial import archive_files,sha
def run(job,out):
    read=lambda p:json.loads(p.read_bytes());scene=out/'scene';ev=read(scene/'execution.json');files={k:job/k for k in ['build_current_cardiac_certificate.py','publish_current_cardiac_certificate.py','current-cardiac-records.json','connected-back-recovery.json']}
    for p in [out/'app/numi-human-native',out/'app/NumiHumanRestingVisual.hpp',out/'build.json']:
        files[str(p.relative_to(out))]=p
    for p in scene.rglob('*'):
        if p.is_file():files['scene/'+str(p.relative_to(scene))]=p
    receipt=job/'current-cardiac-certificate-recovery.json'
    if not receipt.exists():archive_files(files,'current-cardiac-pressure-domain-'+sha(scene/'launch.json')[:15]+'.tar.zst',receipt,'Actual native run with source-equivalent cardiac tube pressure certification and registered tissue mechanics.')
    rc=read(receipt);stage=job/'current-cardiac-certificate-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';e=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in [code,e]:d.mkdir(parents=True,exist_ok=True)
    for k in ['build_current_cardiac_certificate.py','publish_current_cardiac_certificate.py']:shutil.copy2(job/k,code/k)
    logs=(scene/'stderr.log').read_text().splitlines();stdout=(scene/'stdout.log').read_text().splitlines();tr=scene/'native-traces/resting-coupled.csv';states=list(csv.DictReader(tr.open())) if tr.is_file() else []
    markers=('resting_vascular_external_pressure=','resting_tissue_shared_runtime=','myosim_articulated_visual=','articulated_muscle_step_count=','persistent_stand_failure','resting_fem_skin_bed_source=','material_cell_witness ','resting_muscle_step','persistent_stand_progress','physical_step_failure','resting_coupled_')
    record=dict(device='Apple M6',model=read(out/'build.json')['binary'],launch_sha256=sha(scene/'launch.json'),physical_law='Existing seven-coordinate cardiac field retains the certified pressure intervals and GPU acceptance rules.',execution=ev,coupled_trace_rows=len(states),coupled_times_s=[v['time_s'] for v in states],messages=[s for s in stdout+logs if s.startswith(markers)],stderr_tail=logs[-16:],archive={k:rc[k] for k in ['url','archive_sha256','size_bytes']})
    (e/'current-cardiac-pressure-domain-execution.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
