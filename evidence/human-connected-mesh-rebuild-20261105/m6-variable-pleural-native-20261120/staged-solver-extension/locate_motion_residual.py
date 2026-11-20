"""Resolve the compiled owner of the native FEM certificate's residual witness."""
import json
from pathlib import Path
import re

trial = Path("motion-extension")
failure = json.loads((trial / "failure.json").read_text())
log = (trial / "stdout.log").read_text()
assert failure["matter_code"] == 10
residual, correction, dominant_norm, packed_witness = failure["matter_diagnostics"]
tag = -int(packed_witness)
bank, node = tag >> 20, (tag & ((1 << 20) - 1)) - 1
assert bank == 1
objects = [
    (int(object_id), int(first), int(count), int(region_begin), int(region_end))
    for object_id, first, count, region_begin, region_end in re.findall(
        r"^resting_fascia_object=compiled_reference_mode object=(\d+) state_offset=(\d+) "
        r"state_count=(\d+) element_offset=\d+ element_count=\d+ descriptor_flags=\d+ "
        r"region_begin=(\d+) region_end=(\d+)", log, re.M,
    )
]
object_id, first, count, region_begin, region_end = next(
    record for record in objects if record[1] <= node < record[1] + record[2]
)
result = {
    "accepted_steps_before_attempt": failure["segment_start"],
    "matter_microsteps_completed": failure["matter_microsteps"],
    "final_nonlinear_residual": residual,
    "relative_correction": correction,
    "dominant_residual_norm": dominant_norm,
    "residual_bank": "fem_velocity",
    "peak_node_index": node,
    "fascia_object_index": object_id,
    "compiled_fascia_node_range": [first, first + count],
    "source_region_indices": list(range(region_begin, region_end)),
    "local_node": node - first,
}
(trial / "residual-owner.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
