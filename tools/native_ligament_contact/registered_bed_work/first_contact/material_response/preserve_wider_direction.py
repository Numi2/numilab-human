"""Record whether the shared-node solver changed the coupled native rejection at the observed live count."""
from pathlib import Path
import json,re,shutil,sys
from archive_cavity_trial import archive_files,sha
from preserve_contact_registration import native

def run(out):
    trial=out/'incident64-scene';ref=out/'cortical-fit/scene';cfg=json.loads((trial/'launch.json').read_bytes())
    if cfg.get('parent_launch')!=sha(ref/'launch.json'):raise ValueError('Unmatched launch')
    lines=(trial/'stderr.log').read_text().splitlines()
    data=next(l for l in lines if l.startswith('registered_volume_direction_outcomes '))
    if not any(l.startswith('matter_pipeline_compile_end=nm_fem_registered_volume_active_incident64 ') for l in lines):
        raise ValueError('Wider solver was not loaded')
    m=re.search(r'iteration=(\d+) live=(\d+) tile_reason=(.*)$',data)
    if not m:raise ValueError('Missing GPU assignments')
    assignments={int(k):int(v) for k,v in (s.split('/') for s in m[3].split(',') if s)}
    if int(m[2])<=64 and len(assignments)!=int(m[2]):raise ValueError('Incomplete sparse trace')
    source=['build_wider_incident_direction.py','verify_incident_schedule.py','test_incident_schedule.mm','preserve_wider_direction.py']
    files={k:out/k for k in source+['incident64/build.json','incident64/app/build.json','incident64/app/numi-human-native','incident64/runtime/libmetalrobo.dylib','incident64/runtime/runtime.mm','incident64/app/app-runtime.mm','incident64/metal/NumiMatter.metallib','incident64/metal/src/metal/fem.metalinc','schedule64/result.json','schedule64/check.metal','incident-direction-recovery.json']}
    for k in ['launch.json','execution.json','stderr.log','stdout.log','native-traces/resting-coupled.csv','failure.json']:
        if (trial/k).exists():files['incident64-scene/'+k]=trial/k
    receipt=out/'incident64-recovery.json'
    if not receipt.exists():archive_files(files,'registered-incident64-'+sha(trial/'launch.json')[:16]+'.tar.zst',receipt,
        'Matched native shared-node direction over the measured active set. Exact physical runtime, kernel, complete diagnostics and independent GPU scheduling checks.')
    stage=out/'incident64-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response'
    evidence=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261124'
    code.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    for s in source:shutil.copy2(out/s,code/s)
    summary=dict(control=native(ref),coupled=native(trial),gpu_schedule=json.loads((out/'schedule64/result.json').read_bytes()),
        active_tiles=int(m[2]),outcomes_by_tile=assignments,outcome_semantics={'1':'inactive or already converged','2':'insufficient private nodes','3':'fixed node or ineligible mass','4':'starting determinant outside bounds','5':'full proposed endpoint already feasible','6':'direction written and rechecked','7':'refinement failed to improve','8':'post-balance correction failed','9':'singular local metric','10':'refinement exhausted'})
    (evidence/'incident64-volume-direction.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');shutil.copy2(receipt,evidence/receipt.name)
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
