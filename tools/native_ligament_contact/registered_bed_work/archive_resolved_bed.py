"""Recover the exact active contact owner, native returns and independent certificates."""
from pathlib import Path
import json
from archive_cavity_trial import archive_files,sha

def main(case):
    files={};owner=case/'live-owner'
    for k,v in json.loads((owner/'identities.json').read_text()).items():
        if sha(owner/k)!=v:raise ValueError("unmatched binary or shader")
        files['owner/'+k]=owner/k
    names=['identities.json','build-commands.json','shader-build-commands.json']
    for name in names:files['owner/'+name]=owner/name
    for p in owner.iterdir():
        if p.suffix in ('.hpp','.metal','.metalinc','.mm'):files['owner/'+p.name]=p
    for key in ['native-columns','material-32','strict-9','strict-8','entry-8','live-source','column-source']:
        for p in (case/key).glob('*'):
            if p.is_file():files[key+'/'+p.name]=p
    for name in ['registered-columns.csv','resolved-motion-witness.json','live-com-load.json','live-coupon.json',
                 'live-coupon-witness.json','dense-load.json','export_contact_columns.py','build_resolved_bed.py',
                 'NumiHumanResolvedBed.hpp','NumiHumanResolvedBed.metalinc','run_bed_work.py','run_bed_prefix.py',
                 'run-registered-bed','audit_terminal_bed.py','test_resolved_bed.mm','resolve_registered_bed_load.py',
                 'rank_dense_release_load.py','archive_resolved_bed.py']:
        files['work/'+name]=case/name
    for p in [case/'strict-8-source/witness.json',case/'strict-9-source/witness.json']:
        files[str(p.relative_to(case))]=p
    archive_files(files,'resolved-bed-contact-'+sha(owner/'metalrobo/HumanRestingSkin.metallib')[:16]+'.tar.zst',
        case/'recovery.json','M6 native live anatomical force owner, registered lower surface, replay certificates, failed ninth-step material state, matched load control and accepted force transfer test. No claim of settled whole-body motion.')
if __name__=='__main__':main(Path.cwd())
