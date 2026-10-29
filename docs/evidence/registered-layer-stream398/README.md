# Bound anatomical registration scratch to one layer

The registered O/I/D exporter retained all three layers of per-node dictionaries, including 86 body-weight values per vertex, until the export completed. It now writes each layer and releases its row dictionaries before constructing the next. The common parent-registration table is written in the same O/I/D row order. Geometry, nearest-parent selection, exact tie resolution, donor frames, weights and provenance are unchanged.

The parity regression compares all 16 output files against the pinned prior exporter byte for byte. It checks that each prior layer result is released before the next is allocated and that a rejected export leaves no final/staging directory. Existing source-identity/full86 checks and exact parent-registration tests pass; two retained near-tie witnesses and the actual 86-body native matrix capture were also retested. This establishes byte parity and bounded scratch lifetime, not a measured whole-Human runtime speedup.

The full current-surface export uses O295, I288, D375 and the captured parent330 packet. Its authoring output has separate geometry and physics admission flags. Volume meshing, conservative child mass transfer and native motion are not established by these exporter tests. The retained one-step physical parent controls still reject at the material determinant upper limit; this change does not modify that limit, forces or rollback.

The baseline exporter snapshot beside the regression is evidence for exact output comparison. Run the regression with Python plus the existing authoring NumPy/SciPy dependencies; no new framework or public artifact format is introduced.
