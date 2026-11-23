# Forearm material volume repair

A full reconstruction from the registered native parent and two sparse files passes the emitted-coordinate cell checks. All 1,450,187 tetrahedra have positive volume, volume above 1e-18 m³ and dimensionless quality at least 1e-5. The minimum quality is 1.0000014e-5. The minimum volume is 1.0002901e-18 m³. Several cells are close to their thresholds and still require motion testing.

Compared with the previous registered left-forearm candidate, fixed-position connectivity changes reduced 225 failures to 109. Recovering source cells on matching oriented subdomain boundaries reduced that count to 79. A full incident-cell solve reduced it to three. Coordinated vertex updates resolved those three after Float32 emission. The largest additional vertex correction was 97.4 µm. The sparse result replaces 12,572 original cells with 13,960. Every registered surface node and all 14 attachment nodes retain their identity. Anatomical source identity includes material and orientation when matching a source cell.

An independent scan found zero self or mutual crossings of the three physical skin/fat surfaces, zero skin-to-bone pairs and 10,130 skin-to-anatomy pairs. The source starting scene had 10,764. The left brachioradialis has 61 pairs compared with 746 originally. The right remains unchanged at 613.

Keeping original cell masses on deformed retained coordinates violates density bounds in both materials. Fat admits an aggregate source-mass and first-moment allocation. The fixed dermal cavity and the tested wider same-donor neighborhoods fail one or more mass or moment constraints. Those results, positions and topology are archived. A complete per-source route, native deployment and verification through motion remain outstanding.

The independent verifier takes the native parent NPZ, the recovered registered-positions NPZ and material-splice NPZ as positional arguments. It checks source SHA-256, reconstructs Float32 coordinates, checks stable identities and occupancy, then recomputes all cell metrics. The recovery manifest points to the archived inputs and preparation code.
