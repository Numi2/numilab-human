"""Load source-bound native geometry and exact predicates for these experiments."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

run = Path(sys.argv[1]).resolve()
audit = run.parent.parent / 'resting-surface-clearance-001'
sys.path[:0] = [str(audit / 'src'), str(run.parent.parent / 'resting-clearance-and-atrium-005')]
from fast_skin_intersection_audit import install_for_float32

install_for_float32()
a = runpy.run_path(str(audit / 'audit.py'))
clearance = a['clearance']
exact = a['intersections']
receipt = json.loads((run/'native/accepted-geometry/step-16.receipt.json').read_text())
source = (run/'native/resting-human.mrvpack.zst').resolve()
data = subprocess.check_output(['zstd','-dc',str(source)])
assert hashlib.sha256(data).hexdigest() == receipt['base_pack_file_sha256']
poses = {p['body_index']:p for p in receipt['initial_anatomical_registration']['body_poses']}
header, directory = clearance._PACK_HEADER, clearance._PACK_DIRECTORY
magic, version, count, *_ = header.unpack_from(data)
assert (magic,version) == (b'MRVPACK2',2)
s = {key:item for i in range(count)
     if (item := directory.unpack_from(data,header.size+i*directory.size)) and (key := item[0])}
prim, ins = s[4], s[5]

def authenticated_primitives():
    # Filter only missing initial pose rows. No substitute pose is applied.
    expected = []
    import struct
    for i in range(prim[4]):
        _, _, _, instance, semantic, _, _, stable = struct.unpack_from('<8I',data,prim[2]+i*64)
        if (semantic,stable) != a['SKIN'] and semantic not in a['BANK_CLASSES']:
            continue
        _, owner, kind = struct.unpack_from('<3I',data,ins[2]+instance*80+32)
        expected.append((i,kind,owner))
    masked = bytearray(data[:prim[2]+prim[3]])
    for i,kind,owner in expected:
        if kind == 3 and owner not in poses:
            # Leave the immutable source pack intact; the temporary audit view
            # drops banks whose original world transforms were not recorded.
            struct.pack_into('<I',masked,prim[2]+i*64+16,0)
    view = masked + data[len(masked):]
    return a['primitives'](view,poses)

a['BANK_CLASSES'].add(a['SKIN'][0])
banks = {key:(x,f) for key,_,x,f in authenticated_primitives()}
skin, skin_faces = banks.pop(a['SKIN'])
del data
