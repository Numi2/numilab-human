# Physical cell shape at the eighth native step

The accepted state has zero inverted or nonfinite skin and fat cells. All 705,091 skin cells and 743,713 fat cells passed the signed-volume and normalized-shape tests at initialization. At 0.125 ms one skin cell and six fat cells fell below the original shape floor of 1e-5. Two fat cells fell at or below the original volume floor of 1e-18 m³. These measures can overlap. The worst fat shape was 9.82155e-6 and the smallest fat volume was 9.87077e-19 m³.

The terminal observer uses the edges already reconstructed from the accepted reference, high displacement and low displacement, with the actual native tetrahedron and material IDs. It records only two small summaries per material. It runs after the accepted transaction. It adds no force, readback, mesh-copy arena, or per-step simulator. A matched eight-step replay reproduced all 77 physical trace columns, the accepted body and respiratory hashes, terminal fingerprint and displayed vertex buffer.

Two skin/tissue triangle pairs appeared and two cleared between the reference and eighth step. They belong to right flexor digitorum superficialis and right extensor pollicis brevis. Exact triangle IDs are in accepted-cell-quality.json. Equality of aggregate counts masked those changes.

Build the observer against the retained pleural-coupling executable with build_registered_cell_shape_audit.py. The ordinary constructor and terminal FEM acceptance gate remain in force. The new opt-in NUMI_HUMAN_REGISTERED_CELL_QUALITY=1 reports the separate engineering quality result.
