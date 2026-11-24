"""Publish the fitted basis, its conservation checks, and a frozen-basis holdout."""
import json,shutil
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(work):
    volume=work/'witness-material'
    trial=work/'half-step-material-08'
    read=lambda p:json.loads(p.read_bytes())
    full=read(volume/'source-embedding-motion.json')
    held=read(volume/'holdout-motion.json')
    basis=volume/'source-embedded-physical-rows.npz'
    if held['frozen_basis']!=sha(basis) or full['refined']['sparse_matrix']!=sha(basis):
        raise ValueError('Missing independently bound holdout')
    previous=read(work/'witness-archive.json')
    record=dict(physical_parent_revision=(work/'witness-pushed.txt').read_text().strip(),physical_parent_archive=previous['url'],interface_coverage=read(volume/'interface-coverage.json'),measured_embedding=full,held_out=held,scope='Reduced coordinate construction and native full-order reference. This basis has no qualified reduced native advance. The detailed mesh remains the physical reference.')
    field=work/'embedded-boundary-motion.json';field.write_text(json.dumps(record,indent=2)+'\n')
    src=['measure_interface_coverage','enrich_source_embedding','test_source_embedded_motion','verify_embedded_holdout','run_embedded_reference','publish_source_embedding']
    inputs=[field,basis,work/'witness-archive.json',*[work/(key+'.py') for key in src]]
    inputs += [volume/name for name in ('strict-local-source-embedding.json','singular-regional-source-embedding.json','interface-coverage.json','source-embedding-motion.json','holdout-motion.json','holdout-physical-walls.json','holdout-projected-walls.json')]
    inputs += [trial/name for name in ('launch.json','execution.json','stdout.log','stderr.log','failure.json','physical-accepted.node','native-traces/resting-coupled.csv','native-traces/resting-surface-audit.csv')]
    inputs += list((trial/'native-traces/accepted-geometry').glob('*'))
    receipt=work/'embedding-archive.json'
    if not receipt.exists():
        archive_files({str(p.relative_to(work)):p for p in inputs if p.is_file()},'frozen-embedded-boundary-motion-'+sha(field)[:16]+'.tar.zst',receipt,'Source-bound skin/fat embedding matrix, projected inertia checks and matched half-step native trial. Delivery includes failed numerical candidates; a holdout comparison does not qualify native reduction throughput.')
    dst=work/'embedding-delivery'
    tool=dst/'tools/native_ligament_contact/registered_bed_work/connected_material_fold/anatomic_interface'
    ev=dst/'evidence/human-connected-mesh-rebuild-20261105/m6-native-conservative-material-replay'
    for d in (ev,tool):d.mkdir(parents=True,exist_ok=True)
    for k in src:shutil.copy2(work/(k+'.py'),tool/(k+'.py'))
    for p in (field,receipt):shutil.copy2(p,ev/p.name)

if __name__=='__main__':run(Path(__file__).resolve().parent)
