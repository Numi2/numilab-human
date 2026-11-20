"""Select exact old receiver rows from the carrier-capacity dual witnesses."""
import csv
import json
from pathlib import Path

import numpy as np

from candidate import load

_, _, _, material, origins, _, _, old_cells, old_material = load()
kept = np.flatnonzero(origins >= 0)
mapped = np.full(len(old_cells), -1, dtype=np.int32)
mapped[origins[kept]] = kept
with np.load('fine-current-paired-carrier-quota-mass-deficit.npz', allow_pickle=False) as proof:
    wanted = set(proof['source_parent'][proof['carrier_potential'] > .5].tolist())
    group = proof['receiver_group']
    group_potential = proof['group_potential']
previous = Path('fine-paired-cut-receiver-rows.json')
selected = set(json.loads(previous.read_text()))
for role, tag in (('skin', 15), ('fat', 16)):
    with Path(f'fine-paired-{role}-carrier-witnesses.csv').open() as stream:
        for fields in csv.reader(stream):
            donor, receiver = int(fields[5]), int(fields[7])
            if donor not in wanted:
                continue
            row = mapped[receiver]
            if row >= 0 and group[row] >= 0 and group_potential[group[row]] > -.5:
                assert material[row] == old_material[receiver] == tag
                selected.add(int(row))
Path('fine-paired-dual-cut-receiver-rows.json').write_text(json.dumps(sorted(selected))+'\n')
print(json.dumps({'targeted_receivers':len(selected),'saturated_carrier_parents':len(wanted)}))
