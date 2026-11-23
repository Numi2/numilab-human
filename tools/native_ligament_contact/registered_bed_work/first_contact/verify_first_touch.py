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
    verify_registered_endpoint(run, tools, reader, predicate, require_clear=True)
    material=json.loads((run/'material-motion.json').read_text())['material']
    if not material['passed']:
        raise ValueError('Accepted state is outside the registered material envelope')


if __name__=='__main__':verify(*(Path(v).resolve() for v in sys.argv[1:]))
