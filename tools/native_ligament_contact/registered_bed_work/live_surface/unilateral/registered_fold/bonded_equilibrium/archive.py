"""Retain the two loaded states, installed binaries, and independent native results."""
from pathlib import Path
import sys
from archive_cavity_trial import archive_files,sha

def archive(root):
    base=root/'bonded-equilibrium';members={}
    def add(p):members[str(p.relative_to(root))]=p
    for n in ('cavity_mass.py','verify_cavity_mass.py','witness.py','witness.json','admissible.py','relax.py','prepare.py','test_admissible.py','test.json','verify_mass.py','verified-mass.json','layers.json','run_trial.sh','qualify.sh','check_trial.py','archive.py','decision.md','metric-control.json','metric-control.npz','fit.json','preload.npz','preparation.json','current.node','owner/NumiHumanLoadedSkinPoseData.inc','owner/numi-human-native','owner/identities.json','result.json','bed-stop/paired/build.py','bed-stop/paired/build/receipt.json','bed-stop/paired/build/libmetalrobo.dylib','bed-stop/paired/parent/launch.json','bed-stop/paired/owner/numi-human-native','bed-stop/paired/owner/identities.json','bed-stop/build.py','bed-stop/test_gate.mm','bed-stop/gate.json','bed-stop/test-gate','bed-stop/build/NumiMatter.metallib','bed-stop/build/libmetalrobo.dylib','bed-stop/build/receipt.json','bed-stop/parent/launch.json','bed-stop/owner/numi-human-native','bed-stop/owner/identities.json'):add(base/n)
    for r in [base/'three',base/'bed-stop/three',base/'bed-stop/paired/three',base/'bed-stop/paired/accepted-1',base/'bed-stop/paired/accepted-2']:
        if not r.is_dir():continue
        for n in ('stdout.log','stderr.log','execution.json','failure.json','terminal.receipt.json','launch.json','native-traces/resting-coupled.csv','motion.json','verified-endpoint.json','admission.json','endpoint-recovery.json'):
            if (r/n).is_file():add(r/n)
    add(root/'material-direction/recovery.json');add(root/'fold-repair/endpoint-recovery.json')
    archive_files(members,'bonded-equilibrium-'+sha(base/'result.json')[:16]+'.tar.zst',base/'recovery.json','Matched native releases of bonded tissue equilibration and bed admissibility during Newton convergence. Registered geometry, optional solver build, full-scene native traces and source physics retained.')
if __name__=='__main__':archive(Path(sys.argv[1]).resolve())
