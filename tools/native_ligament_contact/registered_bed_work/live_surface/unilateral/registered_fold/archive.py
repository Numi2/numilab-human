"""Archive the exact fold input delta, physical owner and coupled qualification."""
from pathlib import Path
import sys
from archive_cavity_trial import archive_files,sha


def archive(root):
    c=root/'fold-repair';mat=c/'material';files={}
    for p in [c/n for n in ('fit.py','select_coupled_patch.py','route_local_mass.py','build_owner.py',
            'compile_mass.py','verify-prefix.sh','compare_endpoint.py','emit.py','compile-registration.sh','replay.sh','archive.py','source.json','trajectory.json',
            'joint.json','recheck.json','routed-axis-0.npz','admission.json','owner/identities.json','owner/numi-human-native',
            'owner/NumiHumanLoadedSkinPoseData.inc','parent/launch.json')]+[mat/n for n in ('material-identity.json',
            'mass-patch.npz','proposal.npz','reference.node','current.node','mesh/receipt.json','fine-mass-routed/mass-transfer.json',
            'material/registration/receipt.json','material/native-role-verification.json','material/p1/rebuild.json')]:
        files[str(p.relative_to(root))]=p
    for prefix in ('feasible-bed/limiting-row','feasible-bed/minimum','fold-repair/three','integrated-update/three'):
        for path in (root/prefix).rglob('*'):
            if path.is_file() and (path.name in {'launch.json','stdout.log','stderr.log','failure.json','admission.json','execution.json'} or
                path.name=='resting-coupled.csv'):
                files[str(path.relative_to(root))]=path
    for name in ('integrated-update/build.py','integrated-update/build/identities.json','integrated-update/build/reference-update.patch',
        'integrated-update/build/NumiMatter.metallib','feasible-bed/recovery.json','feasible-bed/endpoint-recovery.json'):
        files[name]=root/name
    archive_files(files,'dense-registered-fold-'+sha(c/'three/launch.json')[:16]+'.tar.zst',c/'recovery.json',
        'Exact source-owned skin correction, original and repaired coordinates, native positive-volume and mass admissions, full native owner, coupled run and failed alternatives. The parent receipts restore the unchanged physics.')


if __name__=='__main__':archive(Path(sys.argv[1]).resolve())
