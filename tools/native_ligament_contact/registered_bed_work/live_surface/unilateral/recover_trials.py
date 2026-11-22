"""Verify recovery of both integrated solver alternatives and their measured native behavior."""
import sys
from pathlib import Path
from archive_cavity_trial import archive_files, sha


def recover(root):
    payload={}
    for name,run in [('unilateral','one'),('feasible-bed','three')]:
        d=root/name
        for k in ('contact.metalinc','host.inc','build.py','runtime_patch.py','test_projection.py','check_bed.metal',
                  'check_bed.mm','check-bed','test.metallib','derivative.json','owner/identities.json','owner/numi-human-native',
                  'build/NumiMatter.metallib','build/libmetalrobo.dylib','build/build-pins.json','parent/launch.json'):
            payload[name+'/'+k]=d/k
        for k in ('launch.json','stdout.log','stderr.log','failure.json','execution.json'):
            payload[name+'/'+run+'/'+k]=d/run/k
        for p in (d/run/'native-traces').rglob('*.csv'):
            payload[name+'/'+str(p.relative_to(d))]=p
    payload['parent-recovery.json']=root/'bed-certificate/recovery.json'
    payload['feasible-bed/comparison.json']=root/'feasible-bed/comparison.json'
    for n in ('compare.py','recover_trials.py'):
        payload['unilateral/'+n]=root/'unilateral'/n
    archive_files(payload,'registered-fem-unilateral-bed-'+sha(root/'feasible-bed/comparison.json')[:16]+'.tar.zst',
                  root/'feasible-bed/recovery.json',
                  'Integrated full-body unilateral FE contact and friction, projected globalization variant, '
                  'executables, both libraries, sources, device derivative checks and complete three-root failure. '
                  'The parent receipt identifies the unchanged surface, transferred material mass and other inputs.')


if __name__=="__main__":
    recover(Path(sys.argv[1]).resolve())
