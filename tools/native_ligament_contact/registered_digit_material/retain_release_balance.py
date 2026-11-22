"""Archive exact registered initial coordinates, native launch inputs changed from the material checkpoint and rollback."""
import json
from pathlib import Path
import sys
from archive_cavity_trial import archive_files,sha

def run(root, region, native, fit, output):
    owner=region/'native-owner'
    launcher=json.loads((native/'launch.json').read_text())
    manifest=json.loads((owner/'identities.json').read_text())
    exe=owner/'numi-human-native';override=manifest['loaded_pose_override']
    if (launcher['pins'][str(exe)]!=sha(exe) or sha(exe)!=manifest['binary_sha256']
            or sha(region/'current.node')!=override['sha256'] or override['fit_sha256']!=sha(fit)):
        raise ValueError('Native executable, registration, or patch differs')
    files={n:path for n,path in (
        ('native-owner/numi-human-native',exe),('native-owner/identities.json',owner/'identities.json'),
        ('current.node',region/'current.node'),('native-owner/NumiHumanLoadedSkinPoseData.inc',owner/'NumiHumanLoadedSkinPoseData.inc'),
        ('fit.json',fit),('parent-material-recovery.json',root/'paired-fat-material-recovery.json'),
        ('native-launch.json',native/'launch.json'),('execution.json',native/'execution.json'),
        ('stdout.log',native/'stdout.log'),('stderr.log',native/'stderr.log'))}
    if 'registered_bed_footprint' in manifest:
        loader=owner/'NumiHumanLoadedSkinPose.hpp'
        if sha(loader)!=manifest['registered_bed_footprint']['loader_sha256']:raise ValueError('Unmatched bed footprint code')
        files['native-owner/NumiHumanLoadedSkinPose.hpp']=loader
    receipt=native/('failure.json' if (native/'failure.json').is_file() else 'terminal.receipt.json')
    files[receipt.name]=receipt
    for p in sorted((native/'native-traces').rglob('*.csv')):
        files[str(p.relative_to(native))]=p
    archive_files(files,'bonded-release-'+sha(exe)[:16]+'-'+sha(native/'launch.json')[:12]+'.tar.zst',output,
        'Loaded coordinates and changed files relative to paired-fat material, actual executable, complete native traces and transaction receipt.')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
