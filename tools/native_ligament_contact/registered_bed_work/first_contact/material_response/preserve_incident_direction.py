"""Retain the tested coupled GPU direction and both outcomes of host/scene integration."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha
from preserve_contact_registration import native

def run(out):
    read=lambda f:json.loads(f.read_bytes())
    files={}; code=['build_coupled_volume_direction.py','link_incident_scene.py','test_incident_schedule.mm','verify_incident_schedule.py','run_incident_direction.py','preserve_incident_direction.py','build_unmodified_newton_control.py','trace_incident_outcomes.py','measure_deep_boundary_star.py','bind_observed_runtime.py','measure_initial_bed_support.py','preflight_support_rotation.py']
    for n in code+['contact-registration-recovery.json','incident-direction/build.json','incident-direction/runtime/runtime.mm','incident-direction/runtime/libmetalrobo.dylib','incident-direction/metal/src/metal/fem.metalinc','incident-direction/metal/NumiMatter.metallib','bound-incident-app/build.json','bound-incident-app/app-runtime.mm','bound-incident-app/numi-human-native','schedule-test/check.metal','schedule-test/result.json','incident-scene/launch.json','incident-scene/execution.json','incident-scene/stderr.log']:
        files[n]=out/n
    for n in ['deep-boundary-star.json','initial-support.json','support-rotation.json','incident-observed-owners/build.json','incident-observed-owners/runtime/runtime.mm','incident-observed-owners/runtime/libmetalrobo.dylib','incident-observed-owners/app/app-runtime.mm','incident-observed-owners/app/numi-human-native','incident-observed-owners/app/build.json','unmodified-newton/build.json','unmodified-newton/metal/NumiMatter.metallib','incident-diagnostic/build.json','incident-diagnostic/app/build.json','incident-diagnostic/app/numi-human-native','incident-diagnostic/metal/NumiMatter.metallib','incident-diagnostic/app/app-runtime.mm','incident-diagnostic/metal/src/metal/fem.metalinc','incident-app/build.log']:
        files[n]=out/n
    for name in ['unmodified-scene','incident-observed','observed-shared']:
        for n in ['launch.json','execution.json','stderr.log','failure.json','native-traces/resting-coupled.csv']:
            files[name+'/'+n]=out/name/n
    coupled=out/'incident-paired';control=out/'cortical-fit/scene'
    base=native(control);trial=native(coupled)
    cfg=read(coupled/'launch.json');parent=read(control/'launch.json')
    if sha(control/'launch.json')!=cfg['parent_launch']:raise ValueError('Not a matched parent')
    def same_config(a,b):
        pa,pb=read(a/'launch.json'),read(b/'launch.json')
        if pa['argv'][1:4]+pa['argv'][5:]!=pb['argv'][1:4]+pb['argv'][5:]:return False
        ea,eb=pa['environment'],pb['environment']
        if ea.keys()!=eb.keys():return False
        for k in ea:
            if k in ('DYLD_LIBRARY_PATH','NUMI_MATTER_METALLIB_OVERRIDE'):continue
            if ea[k].replace(str(a)+'/','$run/')!=eb[k].replace(str(b)+'/','$run/'):return False
        return True
    if not all(same_config(control,s) for s in [coupled,out/'unmodified-scene',out/'incident-observed',out/'observed-shared']):raise ValueError('Changed physical configuration')
    actual=read(out/'incident-direction/build.json')
    stderr=(coupled/'stderr.log').read_text()
    if str(out/'incident-direction/runtime/libmetalrobo.dylib') not in stderr or 'matter_pipeline_compile_end=nm_fem_registered_volume_active_incident' not in stderr:
        raise ValueError('Native trial did not reach the integrated GPU owner')
    for n in ['execution.json','stdout.log','stderr.log','launch.json','failure.json','native-traces/resting-coupled.csv']:
        f=coupled/n
        if f.is_file():files['incident-paired/'+n]=f
    rec=out/'incident-direction-recovery.json'
    if not rec.exists():
        archive_files(files,'registered-incident-volume-'+sha(coupled/'launch.json')[:16]+'.tar.zst',rec,
            'Apple M6 shared-node volume-direction test, both host entry points, exact shader/binaries, matched native inputs and corresponding failed startup. Body registration and donor ledger are in parent recovery.')
    staging=out/'incident-delivery';directory=staging/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response'
    evidence=staging/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261124'
    for d in [directory,evidence]:d.mkdir(parents=True,exist_ok=True)
    for n in code:shutil.copy2(out/n,directory/n)
    shutil.copy2(rec,evidence/rec.name)
    info=dict(native_schedule_test=read(out/'schedule-test/result.json'),same_physical_inputs=True,added_scratch_bytes=actual['additional_scratch_bytes'],
              control=base,incident_direction=trial,initial_linkage_error='The statically linked scene runtime still requested the old hint pipeline. The paired build links the same solver into the executable and the shared library.',
              parent=sha(out/'contact-registration-recovery.json'))
    info['unmodified_newton']=native(out/'unmodified-scene')
    info['bounded_gpu_trace']=native(out/'observed-shared')
    info['gpu_cell_direction_outcomes']=[l.strip() for l in (out/'observed-shared/stderr.log').open() if l.startswith('registered_volume_direction_outcomes ')]
    if not info['gpu_cell_direction_outcomes']:raise ValueError('Missing selected GPU direction evidence')
    info['deep_boundary']=read(out/'deep-boundary-star.json')
    info['initial_support']=read(out/'initial-support.json')
    info['support_pose_preflight']=read(out/'support-rotation.json')
    (evidence/'incident-volume-direction.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
