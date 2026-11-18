"""Measure the first rejected chamber state against the exact authored atlas."""
import hashlib
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parent
run = root/'pressure-on'
launch = json.loads((run/'launch.json').read_text())
argv = launch['argv']
receipt = Path(argv[argv.index('--resting-anatomy-receipt')+1])
desc = json.loads(receipt.read_text())['provenance']['cardiac_geometry_binding']['common_field']
witness = json.loads((run/'common-failure.json').read_text())['first_failure_latch']
q = witness['common_first']+witness['common_second'][:3]
sidecar = receipt.parent/desc['domain_boxes']['path']
raw = sidecar.read_bytes()
assert len(raw) == desc['domain_boxes']['count']*64
assert hashlib.sha256(raw).hexdigest() == desc['domain_boxes']['sha256']
ranked = []
for index, box in enumerate(struct.iter_unpack('<16f',raw)):
    deltas = [max(box[i]-q[i],q[i]-box[i+8],0.0) for i in range(7)]
    ranked.append((max(deltas),sum(deltas),index,deltas,list(box[:7]),list(box[8:15])))
ranked.sort()
assert ranked[0][0]>0 and witness['common_status'][0]==5
record = {
    'coordinate_order':desc['coordinate_order'],
    'first_rejected_control_step':witness['step'],
    'last_accepted_time_s':witness['step']*0.002,
    'unpublished_candidate_time_s':(witness['step']+1)*0.002,
    'candidate_blood_volume_targets_m3':witness['chamber_targets_m3'],
    'rejected_geometry_coordinates':q,
    'nearest_box_linf_gap_index_deltas_lower_upper':ranked[0],
    'volume_solve_residual':witness['common_diagnostics'][0],
    'volume_solve_tolerance':desc['solver']['relative_volume_tolerance'],
    'atlas_count':desc['domain_boxes']['count'],
    'atlas_sha256':desc['domain_boxes']['sha256'],
    'geometry_map_sha256':desc['map']['sha256'],
    'volume_polynomials_sha256':desc['polynomials']['sha256'],
}
(root/'atlas-witness.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
