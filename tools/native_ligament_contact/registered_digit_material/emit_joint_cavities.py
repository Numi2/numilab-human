"""Assemble disjoint verified repairs in the existing fixed-cavity schema."""

from pathlib import Path
import sys, tempfile
import numpy as np
from emit_cavity_proposal import emit


def join(native, ledger, out, fits):
    tets = []
    masses = []
    ancestors = []
    retired = set()
    nodes = set()
    counts = [0]
    parents = [0]
    if out.exists():
        raise FileExistsError(out)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        total = None
        identity = None
        for i, fit in enumerate(fits):
            f = Path(tmp) / f"{i}.npz"
            emit(native, fit, ledger, f)
            with np.load(f) as p:
                ne = len(p["replacement_tetrahedra"])
                tt = p["replacement_tetrahedra"]
                nr = p["replacement_parent_offsets"]
                old = p["replacement_parent_cell_rows"][: nr[1]]
                nt = len(p["retained_input_cell_rows"]) + len(old)
                sha = str(p["source_volume_sha256"])
                if total is not None and (nt != total or sha != identity):
                    raise ValueError("Mismatched imported models")
                if set(old) & retired or set(tt.ravel()) & nodes:
                    raise ValueError("Coupled cavity requires a joint local mass solve")
                total = nt
                identity = sha
                retired.update(old.tolist())
                nodes.update(tt.ravel().tolist())
                tets.extend(tt)
                masses.extend(p["replacement_mass_kg"])
                old_ancestry = p["replacement_parent_cell_rows"]
                ancestors.extend(old_ancestry)
                parents.extend((parents[-1] + nr[1:]).tolist())
                counts.append(counts[-1] + ne)
    np.savez_compressed(
        out,
        retained_input_cell_rows=np.setdiff1d(
            np.arange(total, dtype="<u4"), list(retired)
        ).astype("<u4"),
        replacement_tetrahedra=np.array(tets, dtype="<u4"),
        replacement_tet_offsets=np.array(counts, dtype="<u4"),
        replacement_parent_offsets=np.array(parents, dtype="<u4"),
        replacement_parent_cell_rows=np.array(ancestors, dtype="<u4"),
        replacement_mass_kg=np.array(masses, dtype="<f8"),
        source_volume_sha256=np.array(identity, dtype="<U64"),
    )


if __name__ == "__main__":
    join(
        *(Path(x).resolve() for x in sys.argv[1:4]),
        [Path(x).resolve() for x in sys.argv[4:]],
    )
