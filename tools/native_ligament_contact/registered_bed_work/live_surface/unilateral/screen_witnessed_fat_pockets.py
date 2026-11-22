"""Rebuild the native object prefix before testing any local tetrahedral replacement."""
import json
import re
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from locate_material_interface_rejection import audit
from enumerate_material_pocket import screen


def match(case, ledger, library, out):
    failed, accepted = case/'root-two', case/'two'
    text = (failed/'stdout.log').read_text()
    def rows(pattern, count):
        found = re.findall(pattern, text, re.MULTILINE)
        if len(found) != count:
            raise ValueError('Unexpected object inventory')
        return np.asarray(found, dtype=int)
    fascia = rows(r'^resting_fascia_object=source_prepared object=\d+ .*?nodes=(\d+) tetrahedra=(\d+)', 12)
    acl = rows(r'^resting_inferred_acl=admitted .*?nodes=(\d+) tetrahedra=(\d+)', 2)
    outer, wall_faces = rows(r'^resting_aortic_wall_outer_contact=accepted .*?outer_ring_rigid_query_nodes=(\d+) .*?retained_wall_faces=(\d+)', 1)[0]
    prefix=fascia.sum(axis=0)+acl.sum(axis=0)+np.array([2*outer,3*wall_faces])
    actual = rows(r'\bfirst_global_outer_fem_node=(\d+)\b', 1)[0]
    if prefix[0] != actual:
        raise ValueError('Different compiled skin/fat prefix')
    e = json.loads((failed/'launch.json').read_text())['environment']
    _, _, _, _, pins = inputs(failed)
    native = out/'object-offset.json'
    native.write_text(json.dumps(dict(source=pins, source_element_offset=int(prefix[1]), source_node_offset=int(actual)), indent=2)+'\n')
    pockets = out/'material-pockets.json'
    audit(failed, accepted, Path(e['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']), ledger, native, pockets)
    screen(failed, accepted, pockets, ledger, library, out/'admissible-fillings.json')


if __name__=='__main__':
    match(*(Path(s).resolve() for s in sys.argv[1:]))
