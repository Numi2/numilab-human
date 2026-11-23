"""Verify a full-tissue accepted endpoint using the existing registered Human audits."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from material_context import sha
from measure_material_motion import measure
from run_cavity_motion import digest


def verify(run, tools, reader, predicate):
    cfg=json.loads((run/'launch.json').read_text())
    execute=json.loads((run/'execution.json').read_text())
    count=int(cfg['argv'][cfg['argv'].index('--muscle-step-count')+1])
    flags=cfg['environment']
    if not cfg.get('first_contact') or any(flags.get(k)!='1' for k in (
            'NUMI_HUMAN_FEM_SKIN_SUPPORT','NUMI_HUMAN_RESTING_PASSIVE_FASCIA','NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM')):
        raise ValueError('Expected the full-volume first-contact scene')
    if execute['exit']:
        failed=run/'failure.json'
        if failed.is_file():
            detail=json.loads(failed.read_text())
            raise SystemExit(f"Native rejection after {detail['segment_start']} steps, Matter code {detail['matter_code']}, object {detail['matter_object_index']}")
        raise SystemExit(f"Native process exited with status {execute['exit']}")
    geom=run/'native-traces/accepted-geometry'
    path=geom/f'step-{count}.mrvpack'
    receipt=geom/f'step-{count}.receipt.json'
    r=json.loads(receipt.read_text())
    if (execute['exit'] or r['accepted_step']!=count or r['physical_endpoint']!='accepted' or
            r['surface_audit_endpoint']!='passed' or Path(r['accepted_pack_path'])!=path or digest(path)!=r['pack_file_sha256']):
        raise ValueError('Run lacks an authentic native terminal frame')
    archive=run/'accepted-terminal.mrvpack.zst'
    if not archive.exists():
        with archive.with_suffix('.partial').open('wb') as f:
            subprocess.run(['zstd','-q','-3','-T1','-c',str(path)],stdout=f,check=True)
        archive.with_suffix('.partial').rename(archive)
    shutil.copy2(receipt,run/'terminal.receipt.json')
    report=dict(exit_code=0,accepted_steps=count,requested_roots=count,wall_seconds=execute['wall_s'],
                terminal_pack_sha256=sha(path),terminal_archive_sha256=sha(archive))
    (run/'admission.json').write_text(json.dumps(report,indent=2)+'\n')
    measure(run,run/'material-motion.json')
    from check_cavity_endpoint import verify as verify_registered_endpoint
    verify_registered_endpoint(run, tools, reader, predicate)
    endpoint=json.loads((run/'verified-endpoint.json').read_text())
    motion=json.loads((run/'material-motion.json').read_text())
    material=motion['material']
    failures=dict(endpoint['endpoint_anatomy']['failures'])
    for name in ('outside_stretch_envelope','outside_stress_cap'):
        if material[name]:failures[name]=material[name]
    print(f"{count} accepted steps, {r['accepted_time_s']:.9g} simulated s, "
          f"bed reaction {motion['last_bed_reaction_N']:.6g} N, weight {motion['world_weight_N']:.6g} N")
    if failures or not endpoint['endpoint_anatomy']['passed'] or not material['passed']:
        raise SystemExit('Terminal anatomy and material checks failed '+json.dumps(failures,sort_keys=True))
    print('Terminal anatomy and material checks passed')


if __name__=='__main__':verify(*(Path(v).resolve() for v in sys.argv[1:]))
