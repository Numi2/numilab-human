"""Check which repaired cells can retain the current compiler's exact typed labels."""
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

out = Path.cwd()
previous = out.parent / "resting-clearance-and-atrium-011"
sys.path.insert(0,str(previous))
from candidate import COMPILER, load

_, xyz, cells, material, origin, groups, _, _, old_material = load()
with np.load(previous/"fine-mass-local-position-delta.npz",allow_pickle=False) as patch:
    xyz[patch["changed_rows"]] = patch["positions_f32"]
label_file = COMPILER/"volume-export/volume/child.labels.csv"
labels = np.loadtxt(label_file,skiprows=1,dtype=np.int32,delimiter=",")
assert np.array_equal(labels[:,0],np.arange(len(old_material)))
assert np.array_equal(labels[:,1],old_material)
existing = origin>=0
assert np.array_equal(labels[origin[existing],1],material[existing])
counts = collections.Counter()
details = []
for group, ancestors in groups.items():
    selected = np.flatnonzero(origin==group)
    if not len(selected):
        continue
    ids = np.unique(labels[ancestors,1:],axis=0)
    assert len(set(ids[:,0]))==1 and np.all(material[selected]==ids[0,0])
    identical_donor_region = len(np.unique(ids[:,:3],axis=0))==1
    count = len(selected)
    flags = "exact_scalar_label" if len(ids)==1 else (
        "source_face_needs_fractional_rebind" if identical_donor_region else
        "donor_region_needs_fractional_rebind")
    counts[flags] += count
    details.append(dict(cavity=group,reference_ancestry_rows=ancestors,
                        candidate_rows=selected.tolist(),
                        possible_scalar_labels=ids.tolist(),
                        rebind_class=flags))
assert sum(counts.values()) == int((~existing).sum())
with label_file.open('rb') as stream:
    old_label_hash=hashlib.file_digest(stream,'sha256').hexdigest()
report = {
    "geometry_node_count":len(xyz),
    "geometry_cell_count":len(cells),
    "source_label_file_sha256":old_label_hash,
    "emitted_float32_sha256":hashlib.sha256(xyz.astype('<f4',copy=False).tobytes()).hexdigest(),
    "emitted_index_sha256":hashlib.sha256(cells.astype('<u4',copy=False).tobytes()).hexdigest(),
    "cells_with_preserved_source_cell_ancestry":int(existing.sum()),
    "repaired_cells":int((~existing).sum()),
    "counts":dict(counts),
    "cavities":details,
    "installation_status":"ancestry screen only; exact current P1 proofs and native receipt are required",
}
(out/"fine-repaired-semantic-domain.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="cavities"},indent=2))
