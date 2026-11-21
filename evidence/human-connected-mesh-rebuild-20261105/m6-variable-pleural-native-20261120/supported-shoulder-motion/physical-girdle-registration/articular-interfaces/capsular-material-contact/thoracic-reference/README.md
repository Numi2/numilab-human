# Shoulder and connected arm registration

The authoring code derives a shared shoulder reference on SSH Numi from the existing torso. It preserves both humeral scales, the measured source glenoid-facing material gaps, the geometry of the rigid arm segments and the native elbow hinge. The mixed-source assembly still fails physical skin/fat transfer and right capsular contact. The complete native scene continues to use the previously published assets.

## Reference and skeletal clearance

The 22 exact-topology ribs and sternum banks agree on their native body frame to within 0.000052 mm RMS per bank. Three prior rib repairs do not participate in estimating this transform. All original bones remain in the neighboring-bone check. The articular surface fit uses 701 left and 654 right humeral head faces. Two angular holdouts per side have 1.12–1.23 mm RMS radial error. Varying the selected surface changes the inferred center by at most 0.55 mm.

The calibrated native humeri are larger than the common-scale atlas humeri. Centering their heads alone produced 52 left and 37 right intersecting pairs at neutral. The smallest rigid translations preserving source material gaps along the normals of the contiguous glenoid patches are 1.29 mm left and 1.27 mm right. The constraints use 851 and 694 scapular faces, respectively. An exact Float32 triangle audit then found zero paired-bone crossings at 31 distinct configurations per side, covering neutral and rotations of 1–5 degrees in both directions about each atlas axis. Cartilage deformation and dynamic glenohumeral translation were not evaluated.

At the captured native entrance, each humeral rotation and translation carries all 30 descendant bone banks as a unit. The two scapulae move separately. Intersections between these groups and other skeletal banks fell from 526 to zero. The minimum bone-to-bed gap was 39.84 mm. Internal interfaces within each rigidly transported group have identical relative geometry by construction. These counts cover the changed skeletal interfaces at that entrance. They do not test every whole-body anatomical interface or prove support force.

The wrists move freely in this reference. An independent two-link calculation preserving the captured wrist position instead requires at least 55.83 and 56.01 mm elbow relocation at the native entrance, with a freely orientable shoulder and the donor pronation angle. It closes to numerical precision across 255 configurations. The saved orientations are diagnostic target poses, never external forces or joint locks.

## Physical transfer and capsule

| Test at the native entrance | Outer skin pairs | Inner skin pairs | Deep fat pairs | Inverted skin / fat cells |
| --- | ---: | ---: | ---: | ---: |
| Retained volume and skeleton | 0 | 0 | 1,707 | 0 / 0 |
| Connected arms with the retained volume | 465 | 562 | 1,175 | 0 / 0 |
| Connected arms with geodesically blended volume | 63 | 325 | 2,511 | 1,948 / 1,831 |

The table counts intersections of the 62 changed bone banks against the three layer boundaries. The physical screen uses all 311,020 nodes and 1,448,804 tetrahedra. Each of the 118,196 repeated interface rows has the same physical node coordinates and donor on both partitions. It contributes once to transfer. The geodesic blend runs along material edges, with a transition half-width of one fitted humeral-head diameter. Binary32 emission and the existing signed-volume and scaled-determinant limits are checked over every cell. The candidate fails both material regions. Of its 3,779 inversions, 3,425 occur on cells whose four original nodes belong to the same donor group. Just 21 inversions occur on source cells with an edge exceeding 100 mm. Long edges or donor transitions alone would miss most failures.

A separate Euclidean volume-weighted trial produced 30 outer, 660 inner and 4,006 deep pairs, with 479 inverted skin and 441 inverted fat cells. The preserved rejection record includes the emitted coordinate hash. Neither trial changes the native volume, mass map or elastic rest configuration. A replacement must carry the three boundaries and the interior together, enforce positive cell geometry and source mass during registration, and pass contact before it can take over from the existing physical mesh.

The capsular test uses the same spacing and real scapular and humeral attachment features. Each record is independently reconstructed on the paired bone meshes. The validator checks solid crossings, local unilateral reactions and 27 neighboring source banks, including both raw clavicles, sternum and bilateral ribs. The left route passed 61 of 61 prescribed positions. The right route passed 34 positions and failed at the 35th, a −4° rotation about atlas axis 1. It cleared the bones but required a local reaction outside the admissible normal cone by 0.00978 per unit tension. This is a prescribed component motion test without a human force trajectory. A bilateral native constitutive route remains unresolved.

## Reproduction

Run `./reproduce.sh` from this directory on SSH Numi with the retained inputs. It verifies the shared reference, both humeral head fits, the actual closed-surface spacing, the connected articulated geometry and registered skin/fat cells. It also runs the new authoring unit tests. Use the [preceding native reproduction](../reproduce.sh) for the exact complete-scene build, launch and pre-step rejection. The Human compiler owns the reference calculations in `src/numilab_human/shoulder_registration.py`. All runtime forces and accepted state remain with their existing owners.

BodyParts3D © The Database Center for Life Science, CC Attribution–Share Alike 2.1 Japan. The retained archive and member identities and MyoSim source provide the provenance for this reference adult.
