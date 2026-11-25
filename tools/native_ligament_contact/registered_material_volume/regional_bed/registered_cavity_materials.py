"""Resolve each rebuilt tetrahedron to its original constitutive material."""

import numpy as np


def apply_registered_cavities(cells, labels, cavity):
    keep = cavity["retained_input_cell_rows"]
    new = cavity["replacement_tetrahedra"]
    parent = cavity["replacement_parent_cell_rows"]
    offsets = cavity["replacement_parent_offsets"]
    if any(a.dtype.kind not in "iu" for a in (keep, new, parent, offsets)):
        raise ValueError("Non-integral cavity topology")
    if (cells.shape != (len(labels), 4) or labels.ndim != 1 or
            keep.ndim != 1 or new.shape != (len(new), 4) or parent.ndim != 1 or
            offsets.shape != (len(new) + 1,) or offsets[0] != 0 or
            offsets[-1] != len(parent) or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("Incomplete cavity parent ranges")
    if (np.any(keep < 0) or np.any(keep >= len(labels)) or
            np.any(parent < 0) or np.any(parent >= len(labels))):
        raise ValueError("Cavity row outside original material")
    accounted = np.zeros(len(labels), dtype=bool)
    accounted[keep] = True
    if accounted.sum() != len(keep) or accounted[parent].any():
        raise ValueError("Repeated retained cell or retained cavity parent")
    accounted[parent] = True
    if not accounted.all():
        raise ValueError("Original material cell missing from ancestry")
    owners = labels[parent]
    material = owners[offsets[:-1]]
    if not np.array_equal(owners, np.repeat(material, np.diff(offsets).astype(np.intp))):
        raise ValueError("Replacement spans different tissue materials")
    return np.vstack((cells[keep], new)), np.r_[labels[keep], material]
