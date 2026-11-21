# Gluteal material registration through supported native motion

The left and right gluteus maximus now share a neutral material registration between their pelvic origins and femoral insertions. The previous static surface repair cleared both muscles at the start, but their femoral and pelvic donor frames disagreed. The resulting muscles crossed as soon as the hips moved. In the matched supported control the first observation at 64 ms had eight intersecting pairs, the 16-second endpoint had 267, and the peak was 271.

The updated NHTISS4 keeps the registered entrance positions and normals, source triangles, laterality, anatomical body IDs and existing MyoSim routes. It fits continuous weights between the known origin and insertion neighborhoods. Each donor independently reproduces the same neutral location before the body moves. A constrained fit bounds the material deformation gradient over the actual tetrahedra during release and settling. It uses 33 pose frames, including the first second and spaced observations of the remainder. The separate volume and quality pass checks 64 poses and made zero further weight edits on either side. An independent Float32 intersection and volume check covers 251 distinct registered poses at 64 ms spacing including the entrance.

## Native results on SSH Numi

| Measure | Supported body with 32 physical bed sections | Full skin/fat volume body |
| --- | ---: | ---: |
| Accepted physical time | 16.00 s | 0.125 ms |
| Accepted roots | 8,000 | 8 |
| Launch through exit | 164.808 s | 246.712 s |
| End-to-end factor | 0.09708× | 5.07e-07× |
| Unchanged coupled trace columns | 77 | 77 |
| Intersection pairs between the two glutes at the recorded endpoints | 0 | 0 |

The supported trial presented the native viewer. It recorded 3 breaths and 19 filling/ejection cycles, with pleural pressure between -770.0 and -470.3 Pa. The bed supplied 952.73 N over the last two seconds and the reported peak root assistance was 0 N and 0 N·m. Viewer-inclusive motion after startup measured 0.10493×. Native presentation callbacks averaged 1.53 per wall-clock second.

Both supported native packs agree with the independently replayed registered coordinates within 0.194 µm. Neither self-intersections nor bilateral intersections appeared at any sampled pose. The minimum normalized determinant was 2.76359e-05 across the two meshes. A passive skeletal muscle specimen comparison used the registered entrance as reference, the existing compressible neo-Hookean energy and the conservative first Piola stress Frobenius norm. All 7,161 candidate gluteal tetrahedra passed its volume, energy and stress bounds at every sampled pose. Across both sides, relative cell volume ranged from 0.344 to 1.674. Principal stretches ranged from 0.359 to 1.645. Whole muscle volumes still decreased by as much as 9.5%. The specimen law is uncalibrated. The native assets remain kinematically deformed registered muscle surfaces. These comparison stresses and energies supply no extra forces to the body.

The first post-rebind trial enforced positive cell volumes. Four cells expanded more than tenfold. Adding an upper determinant bound alone left large shear, including principal stretches above 19 and first Piola stress above 9 MPa. The retained deformation-gradient fit addresses both. The failed launches, nonlinear checks, spatial gradient experiment and sparse-pose fit are in `failed/`.

## Remaining physical failures

The full scene contains 318,226 FEM nodes and 157 rigid bodies and retains its 26 fascia contact regions and 34 registered ligament rows. At its endpoint it still has 112 right and 127 left glute-to-deep-fat crossing pairs and 6 against the left inner skin. The sacrum and several other structures also intersect the glutes. All endpoint witnesses are retained. The reduced model has a different skin/fat owner and cannot certify these interfaces. The full-volume release does not establish loaded bed equilibrium or long-duration motion.

Using actual supported row impulses gave a peak mean nominal pressure of 32.85 kPa. Under the full source tissue laws, five rows violate the fat volume limit if transverse strain is suppressed. Three of these rows attach to the pelvis. The late mean force on violating rows was 398.7 N. Allowing free lateral expansion kept the calculated volume ratio above the gate. These two offline boundary assumptions use the same observed forces. The live condensed sections retain their own material parameters. Transverse deformation and posterior load transfer need to be resolved in the full body.

## Replay inputs

Candidate payload SHA-256 is `e3e189d0c69412a985aa75663a3959b95ad98a4b8027f5e7d050f86ef2eb454c`. The full and supported launch configurations, solver binaries, all parent assets and native receipts are hash-pinned through their respective launch records. Both source tetrahedral connectivities are included with the final source-coordinate and binding edits. Use the authoring helpers in `tools/native_ligament_contact/posterior_registration` with the existing material and native evidence modules on the Python path. Run `common_bind_gluteus.py`, `bound_pose_strain.py`, `condition_motion_weights.py`, `emit_neutral_posterior.py`, and `configure_supported_posterior.py` in that order. The compiled candidate is the same content in both launches. The full native packs are recoverable from the stated base hashes and the two lossless deltas.

This is a mixed-source reference adult. The muscle assets adapt BodyParts3D, copyright © 2008 The Database Center for Life Science, under [CC Attribution-Share Alike 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/).

On the retained Numi installation, launch the supported native viewer with `ssh numi '~/bin/numi-human-resting 16'`. The launcher verifies the source and binary, records each new session in its own directory and uses the existing coupled solver. The recovery copy of the launcher is in `scripts/run_resting_human_on_numi.sh`.
