"""Package the new registration and the exact Human owner against the existing recovery chain."""
import json
import sys
from pathlib import Path
from archive_cavity_trial import archive_files, sha

def retain(case,run,out):
    job=json.loads((run/'launch.json').read_text())
    if job['argv'][0]!=str(case/'native-owner/numi-human-native') or job['pins'][job['argv'][0]]!=sha(Path(job['argv'][0])):
        raise ValueError('Different native owner')
    files={}
    for folder in ('material','mesh','fine-mass-routed','qualified-parent'):
        for p in (case/folder).rglob('*'):
            if p.is_file() and p.suffix not in ('.log',):files[str(p.relative_to(case))]=p
    for name in ('native-owner/numi-human-native','native-owner/NumiHumanLoadedSkinPoseData.inc','native-owner/identities.json',
                 'current.node','proposal.npz','native.volumes','material-identity.json','material-bind.json'):
        files[name]=case/name
    ref=case/'mesh/mesh.node'
    if sha(ref)!=sha(case/'reference.node'):raise ValueError('Duplicate reference receipt differs')
    for key,pin in job['pins'].items():
        p=Path(key)
        if p.is_relative_to(case) and (sha(p)!=pin or str(p.relative_to(case)) not in files):
            raise ValueError('Missing runtime input '+key)
    archive_files(files,'paired-fat-material-'+sha(ref)[:16]+'.tar.zst',out,
                  'Exact rebuilt physical registration, role and mass files, native executable and its generated material header. The rest of the owner and shared source inputs inherit the pinned three-step checkpoint.')
if __name__=='__main__':retain(*(Path(a).resolve() for a in sys.argv[1:]))
