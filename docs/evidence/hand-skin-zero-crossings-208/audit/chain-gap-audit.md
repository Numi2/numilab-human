# Envelope 033 chain and gap audit

Read-only review of candidate033 SHA-256 549fab40b3cdfec2d0e1c9a004451d276f87b445f1c7dc9d5a56c1b5a1a53be4. Gap occupancy uses envelope031 as explicit proxy: envelope033 differs at 14 seam vertices by at most 0.995705 um.

The 94-vertex/184-face patch has PCA axis [0.181414, 0.976298, -0.118028], centroid [-307.121, -985.328, 139.736] mm, bounds [-315.335, -1014.699, 133.672] to [-298.484, -960.609, 146.014] mm, and axial span -30.568 to +25.260 mm.

## Geometric chain

The patch follows serial chain 61/97 -> 60/96 -> 59/95 -> 57/89. The 1 mm endpoint-band gaps are 1.434, 1.176, and 1.762 mm; absolute axis dot products are 0.9958, 0.9973, and 0.999895. The patch covers the distal end of 61/97, all of 60/96, and part of 59/95, ending about 11.5 mm before 57/89. Patch-to-surface distances are 0.45-0.49 mm to the first three banks and 11.65 mm to 57/89.

The nearby parallel chain is 66/100 -> 65/99 -> 64/98. Patch distances are 18.44, 11.15, and 11.34 mm, respectively. Available identifiers are body/stable-bone IDs; no anatomical member-name mapping was established. This is a geometric chain inference only.

## Inter-digit gap

At 25 one-millimetre slabs from -25 to -1 mm, four central chord samples (t=0.3, 0.4, 0.5, 0.6) were outside for baseline806 and envelopes006, 014, and 031 at every slab. Saved nine-sample occupancy strings show outer-gap expansion: at s=-20 mm, baseline was 000000000 versus 100000001 for 006/014/031; at s=-14 and -10 mm, 031 was 110000001 versus 100000001 for 006/014 and all-zero baseline; at s=-2 mm, 031 was 110000011 versus 100000011 for 006/014 and all-zero baseline. At selected s=-25 and -22 mm, baseline was 000000011 / 000000111 and 031 was 110000011 / 110000111.

This indicates about 2.5-5.2 mm added expansion toward the inter-digit gap at sampled outer points, while the central sampled gap remained outside. Per-sample XYZ rows were not retained and are not reconstructed; actual patch and bank endpoint coordinates are recorded in the JSON. The finite 1 mm / nine-point sweep does not prove continuous 3D clearance. Envelope031 is a proxy for 033 with the stated <=0.996 um seam difference.

## Status

The exact ordered gate is recorded as PASS_REFERENCE_GEOMETRY_ONLY. Anatomy remains unreviewed and mass is NOT ADMITTED. No physical steps were run.
