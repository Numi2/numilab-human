# Registered mass inside the pelvis

In the accepted full human, the two nonoverlapping pelvic cortical solids contain **2.9154 mL of dermis and subcutaneous fat carrying 2.5542 g**. The result reproduces to the displayed precision in the public joint-stops scene, the forearm control and the wrist candidate. It identifies an anatomical error in the scene used for supported contact. It measures a simulated reference adult at one accepted instant.

| Cortical bank | Dermis volume | Dermis mass | Fat volume | Fat mass |
| --- | ---: | ---: | ---: | ---: |
| 18 | 0.04896 mL | 0.04410 g | 1.58021 mL | 1.36651 g |
| 19 | 0.02748 mL | 0.02514 g | 1.25874 mL | 1.11847 g |

The owner audit reconstructs the accepted tetrahedra and registered bone poses from the same transaction. It computes aggregate intersections of the oriented material boundary with each closed cortex using CGAL. Independently, it clips individual tetrahedra against the solid, including completely enclosed cells, and calculates affected masses from their existing allocations. The per-cell sums must reproduce the aggregate volume. Each ledger row retains its native cell, four parent nodes, overlap fraction, mass and barycentric centre of the displaced material.

**The repair must move material outside the cortex while preserving its volume and anatomical ownership.** A subtraction would discard the mass above. We tested whether the remaining volume could carry it by redistributing density within the same material, donor and anatomical region. Four of the seven affected groups admit the original mass and first moments in both the stress-free and opening frames. Three have a verified separating inequality even with a 50 mm halo around the banks and freedom to redistribute throughout that domain. Those groups carry approximately 0.238 g of the displaced mass. The inequalities rule out that fixed-geometry redistribution under the existing density bounds. They do not rule out changing the shape of the adjacent tissue.

At the deepest pelvic witness the actual native penalty contact allows 3.575 mm of interpenetration. Reconstruction needs a common fat–cortex interface, positive skin/fat cells and room in the adjacent soft tissue for the displaced material. The reference and supported coordinates must be rebuilt together before a new force-transfer or motion result can be claimed. All three audited scenes still use the previous pelvic geometry.

Run the reusable audit on SSH Numi with the native preparation Python environment and the existing geometry dependencies. Set RUN to the native run directory and OUT to a new output directory.

```sh
python tools/native_ligament_contact/registered_material_volume/audit_pelvic_material_overlap.py "$RUN" "$OUT"
```

[Measurements and conservative feasibility certificates](pelvic-material-mass.json) · [Verified recovery of exact ledgers and experiment sources](pelvic-material-mass-recovery.json) · [Matched native motion and support](right-wrist-material.md)
