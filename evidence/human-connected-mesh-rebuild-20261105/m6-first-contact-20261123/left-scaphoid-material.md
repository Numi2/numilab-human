# The physical wrist at release

On SSH Numi, the existing certified source mesh places 70 dermal and 397 fat tetrahedra partly or entirely inside the left scaphoid (FJ3278, stable ID 81). All 12 fat cells failing at the first measured motion overlap this bone. Ten lie wholly inside it. The overlap is 2.15774145 mm³ in skin and 16.12300614 mm³ in fat. Of the 24 nodes in the motion witnesses, 23 are inside this scaphoid at release. An accepted native force capture records a node speed of 0.68 m/s after 7.8125 µs.

## Replacement material

The reconstructed cavity joins both materials and the cortical surface. All 546 triangular faces adjoining the original volume match. The Float32 cavity contains 713 dermal and 1,207 fat cells. The closed-solid intersection with the scaphoid measures 0 mm³ for both materials. The measured skin–fat solid overlap is 3.5 × 10⁻¹⁸ mm³. Outward rounding during authoring moved a contact point by at most 0.081 µm.

Existing physical quadrature allocations were routed through 10,551 positive same-material parent–child volume intersections. Separate equalities preserve frozen allocations. The remaining allocations can move only between destinations reached by their certified upstream quadrature. All density bounds pass. Across both materials the largest source relative error is below 10⁻¹³, mass error below 3 × 10⁻¹⁷ kg, and first-moment component error below 10⁻¹² kg·m. Geometric donor labels follow largest overlap. The archived transport rows retain all participating mass sources.

The unchanged native importer read the full assembled body material (311,441 nodes, 1,450,476 cells). It reported zero inverted, undersized, low-shape and nonfinite cells, zero density outliers, zero massless nodes, and zero discrepancy between independently accumulated nodal masses. The original physical density gates rejected nine fat cells in the first trial. Guarding the optimization inside those bounds resolved the discrepancy in the independent native arithmetic.

## Motion admission

This is a static material import. The repair retires 85 old mechanical nodes, including 23 bonded inner-skin and 52 deep-fat registered nodes. The remaining ten are volume nodes. The native scene needs current surface correspondence, attachment rows and conservative force transfer for the changed topology.

The left scaphoid and capitate also intersect each other by 2.20 mm³ in the loaded pose. The right pair intersects by 4.93 mm³. The original atlas contains overlapping bone labels too. Positioning the left scaphoid in the capitate's atlas frame alone produces 76.0 mm³ of scaphoid–lunate overlap. A local coordinate edit that clears one bank cannot qualify the articulated wrist. Keep the candidate available for coupled registration and motion checks.

The original full-scene control had 10,130 skin-to-interior triangle-pair crossings at release, chiefly against registered muscle. After two accepted substeps, 10,128 identical pairs persisted and the total was 10,132. This points to initial registration as the dominant source. That control advanced 15.625 µs in 189.65 wall seconds including startup. The replacement material has no measured full-scene realtime factor or 16-second motion certificate.

## Recovery and replay

The two adjacent recovery receipts give independent GitHub downloads and individual file hashes. The geometry archive holds the exact construction programs, candidate inputs and meshes, the native witness binary, and both accepted and rejected evidence. The material archive holds the full TetGen geometry, origin-to-child overlap, conservative allocation, compiled-import command and output hashes. The retained physical control is commit 0b07d82a19e7e2f4c02e84d1647533b4cf81ab8a.

From the restored geometry workspace, the owner transfer takes the native control, original material, emitted candidate, parent overlap, eligible quadrature changes, and full carrier flow as inputs. The importer then consumes the resulting routes and the emitted geometry receipt.

```sh
python route_emitted_wrist_material.py "$control" "$original_material" clear-cavity/domain.npz child-overlap wrist-allocation "$full_flow" guarded-mass
python import_wrist_cavity.py "$original_material" clear-cavity/domain.npz child-overlap guarded-mass clear-cavity/solid.json guarded-native
```
