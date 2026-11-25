"""Encode movements at rejecting FEM cells against the compiler's stress-free reference."""

import numpy as np


def encode_incident_displacements(reference, previous, proposed, cells, failed, margins):
    """Ask the native interval evaluator for all cell margins after each local edit.

    The caller supplies the native determinant witnesses and performs the subsequent
    shape, strain, attachment, contact and mass checks on the returned coordinates.
    """
    if previous.shape != proposed.shape or reference.shape != proposed.shape:
        raise ValueError('Material point tables differ')
    if proposed.dtype != np.float64 or not np.isfinite(reference).all():
        raise ValueError('Invalid loaded material precision or reference coordinates')
    if not np.array_equal(reference.astype(np.float32).astype(np.float64), reference):
        raise ValueError('The executable FEM reference must be exactly representable')
    if not np.isfinite(previous).all() or not np.isfinite(proposed).all():
        raise ValueError('Nonfinite material coordinates')
    pending = np.asarray(failed)
    if pending.ndim != 1 or (pending.size and (not np.issubdtype(pending.dtype, np.integer)
            or pending.min() < 0 or pending.max() >= len(cells))):
        raise ValueError('Invalid native cell witnesses')
    output = proposed.copy()
    moved = (previous != proposed).any(axis=1)
    encoded = np.zeros(len(output), dtype=bool)
    history = []
    def evaluate():
        current = np.asarray(margins(output))
        if current.shape != (len(cells),) or not np.isfinite(current).all():
            raise ValueError('Incomplete executable determinant certificate')
        return current
    if not pending.size:
        pending = np.flatnonzero(evaluate() < 0)
    while pending.size:
        nodes = np.unique(cells[pending])
        if nodes.min() < 0 or nodes.max() >= len(output):
            raise ValueError('Cell indexes into a different material')
        nodes = nodes[moved[nodes] & ~encoded[nodes]]
        if not nodes.size:
            raise ValueError('Single-word displacement cannot certify the remaining cells')
        output[nodes] = reference[nodes] + (proposed[nodes] - reference[nodes]).astype(np.float32).astype(np.float64)
        encoded[nodes] = True
        current = evaluate()
        pending = np.flatnonzero(current < 0)
        history.append(dict(nodes=int(encoded.sum()), rejected=len(pending), minimum_margin=float(current.min())))
    return output, history
