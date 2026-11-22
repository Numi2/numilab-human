"""Preserve the intact fitted body, native solver, correlated frame evidence, and original mapping."""
from pathlib import Path
import sys,json
from archive_cavity_trial import archive_files,sha

def archive(case):
    d=case/'free-interface';members={}
    def add(p):members[str(p.relative_to(case))]=p
    for f in ('admissible.py','cavity_mass.py','relax.py','prepare.py','verify_cavity_mass.py','witness.py','check_trial.py','observe_release.py','capture_opening.py','compare_opening.py','crossing_motion.py','opening.py','qualify_interface.sh','archive_interface.py','recover_closed_frames.py','cold-frame-recovery.json','recovery.json'):
        add(case/f)
    for f in ('fit.json','preload.npz','preparation.json','current.node','layers.json','verified-mass.json','uncompensated/fit.json','uncompensated/preload.npz','owner/identities.json','owner/numi-human-native','owner/NumiHumanLoadedSkinPoseData.inc','observed-owner/identities.json','observed-owner/numi-human-native','audit_registered_initial_envelope.py','opening-recovery.json','registration-to-paired2.json','crossings-motion.json','result.json'):
        add(d/f)
    for f in ('six-summary.json','executed-parity.json','executed-envelope/initial-envelope.json','executed-envelope/physical-banks.json','executed-envelope/target-witnesses.jsonl','executed-to-three.json','initial-bank-displacement.json','release-witness.json'):
        if (d/f).is_file():add(d/f)
    for r in (d/'three',d/'accepted-2',d/'six',d/'accepted-4',d/'accepted-5'):
        if not r.is_dir():continue
        for n in ('launch.json','execution.json','admission.json','stdout.log','stderr.log','failure.json','terminal.receipt.json','native-initial-bodies.json','native-traces/resting-coupled.csv','motion.json','verified-endpoint.json','endpoint-recovery.json'):
            if (r/n).exists():add(r/n)
    manifest='-'.join(sha(p)[:9] for p in [d/'result.json',d/'crossings-motion.json']+[d/f for f in ('six-summary.json','executed-to-three.json') if (d/f).is_file()])
    receipt=d/'recovery.json'
    archive_files(members,'registered-loaded-interface-'+manifest+'.tar.zst',receipt,'Full native fitted skin and fat release including source mass, linked native owners, established exact crossing counts and independently restored physical endpoints.')
if __name__=='__main__':archive(Path(sys.argv[1]).resolve())
