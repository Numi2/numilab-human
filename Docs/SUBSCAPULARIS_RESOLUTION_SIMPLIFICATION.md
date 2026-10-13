# Source-bounded subscapularis resolution simplification

`collapse_subscapularis_resolution_edges` applies the existing closed-surface
midpoint-collapse owner to explicitly selected disjoint edges. Inputs retain their
initial source vertex IDs even after previous collapses compact the vertex table.
Attachment/core locks use those source IDs, not the current compacted row number.

The caller supplies finite six-column vertices, integer triangle indices and
face-parent origins, existing canonical body-weight columns, a declared positive
source resolution in metres, and any locked source vertex IDs. Edges longer than
that resolution are rejected. The existing owner still enforces closed manifold,
orientation, endpoint displacement and volume-change admission. Face-parent
lineage is preserved; midpoint body maps are arithmetic averages with no weight
renormalization or reassignment to a new body owner.

```python
from numilab_human.subscapularis_resolution_reconstruction import (
    collapse_subscapularis_resolution_edges,
)

vertices, faces, face_origins, body_maps, audit = (
    collapse_subscapularis_resolution_edges(
        source_vertices6, source_faces, source_face_origins, source_body_maps,
        selected_initial_source_edges,
        resolution_m=declared_source_resolution_m,
        locked_source_vertex_ids=attachment_and_core_source_ids,
    )
)
```

Malformed or lossy source IDs are rejected rather than rounded into another row.
A failure after an earlier accepted local collapse does not mutate caller source
arrays. Audit rows name initial selected edges, merged source IDs, inherited
geometric admission and the declared bounds.

Validation includes two disjoint collapses through compaction, a later locked
attachment, loss of integer source identity, unchanged input arrays on success and
failure, and existing closed-surface owner regressions: 15 tests pass. These are
computational mesh/property checks over declared fixtures. They do not establish
anatomical reconstruction accuracy, source registration, an attachment footprint,
biomechanics or runtime-payload admission; those owners must qualify a real source
independently before consuming the result.
