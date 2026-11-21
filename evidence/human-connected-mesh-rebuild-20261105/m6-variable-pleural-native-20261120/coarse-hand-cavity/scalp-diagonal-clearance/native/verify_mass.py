"""Check emitted cells, registration banks and nodal mass on the native device."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import digest, rows
from cavity import LABEL, determinant


def check(run, control, deck):
    name = "shell-mass.csv"
    paths = [run / (name + suffix) for suffix in (".nodes.csv",".tets.csv",".labels.csv")]
    node, cell, label = (rows(p) for p in paths)
    old_node, _, old_label = (rows(control / p.name) for p in paths)
    fixed = ("node_frame_owner_body",) + tuple(f"{kind}_{a}_m" for a in "xyz"
                                              for kind in ("current","rest","donor_local"))
    assert [tuple(r[k] for k in fixed) for r in node] == [tuple(r[k] for k in fixed) for r in old_node]
    typed = [tuple(r[k] for k in LABEL) for r in label]
    previous = [tuple(r[k] for k in LABEL) for r in old_label]
    assert len(typed) == len(previous)
    actual = np.array([[int(r[f"parent_node{i}"]) for i in range(4)] for r in cell])
    rest = np.array([[float(r[f"rest_{a}_m"]) for a in "xyz"] for r in node])
    actual_mass = np.array([float(r["parent_mass_kg"]) for r in cell])
    emitted_node_mass = np.array([float(r["compiled_lumped_mass_kg"]) for r in node])
    with np.load(deck) as data:
        cells, mass, changed = (data[k] for k in ("cells","cell_mass_kg","changed_rows"))
        np.testing.assert_array_equal(np.sort(actual,axis=1),np.sort(cells,axis=1))
        np.testing.assert_allclose(actual_mass,mass.astype(np.float32),rtol=2.e-6,atol=0.)
        expected_node = np.bincount(cells.ravel(),weights=np.repeat(mass/4.,4),minlength=len(node))
        np.testing.assert_allclose(emitted_node_mass,expected_node.astype(np.float32),rtol=2.e-6,atol=0.)
    mismatches = [i for i, (a, b) in enumerate(zip(typed,previous,strict=True)) if a != b]
    determinant_values = determinant(rest,actual)
    assert np.min(determinant_values) > 0.
    result = {"hashes": {p.name:digest(p) for p in (*paths,deck)},
              "volume_cells":len(cells),"nodes":len(node),"emitted_geometry_matches":True,
              "emitted_mapped_tet_mass_kg":float(actual_mass.sum()),
              "emitted_mapped_node_mass_kg":float(emitted_node_mass.sum()),
              "maximum_cell_mass_error_kg":float(np.abs(mass-actual_mass).max()),
              "maximum_node_mass_error_kg":float(np.abs(expected_node-emitted_node_mass).max()),
              "changed_tetrahedra":len(changed),"min_signed_determinant_m3":float(determinant_values.min()),
              "changed_typed_cell_rows":len(mismatches),
              "first_typed_label_changes":[[i,previous[i],typed[i]] for i in mismatches[:20]]}
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("run","control","deck","output"):
        parser.add_argument(key,type=Path)
    a = parser.parse_args()
    report = check(a.run,a.control,a.deck)
    with a.output.open("x") as outlet:
        json.dump(report,outlet,indent=2,allow_nan=False)
        outlet.write("\n")
    print(json.dumps(report))
    raise SystemExit(bool(report["changed_typed_cell_rows"]))
