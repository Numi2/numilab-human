# Post-dynamics pelvic contact on Apple M6

At the previous eight-step endpoint, tissue publication and freshly reconstructed skeletal poses disagreed by up to 13.124 µm in relative position. The compensated forward-kinematics refresh in the articulated runtime ran only for the dedicated Human/Matter program. The resting scene uses the transaction callback. Its tissue certificate and accepted tissue capture could therefore use the previous skeletal configuration. Earlier counts and material results at those captures describe their captured geometry. They cannot qualify their corresponding accepted skeletons.

The [runtime fix and regression](https://github.com/Numi2/numi-lab/commit/4138685d78e22a6b15826b968dc68cfd5c095790) makes the compensated transaction refresh the borrowed body and point geometry after Stand integrates joint coordinates. One owner advances the body. A moving native regression exposed a quaternion component error of 0.000037 with the original library. The repaired library passed it and the existing transaction, rollback, ABI, candidate FK, mass and staged-force cases. The A/B libraries used the same source distribution, object inputs and native compiler options except for the endpoint refresh. The dynamic loader records identify the actual library in each human run.

The Human contact solver now restores the last accepted contact pressures as an initial guess. It rebuilds the current finite-strain forces, moving surface geometry, pressure compliance, free-node residuals and clearance certificate before reusing them. A loaded row that loses its contact normal fails certification. The velocity, load law, volume ownership and accepted-state transaction use their existing owners. Across 29 corrected steps, 56 of the 58 tissue-side pre-dynamics evaluations passed with zero nonlinear refreshes. All 58 post-dynamics evaluations performed a fresh nonlinear update on the moving skeleton.

| Native observation | Original library | Corrected library |
|---|---:|---:|
| Step 8 maximum relative bone-position error | 13.124 µm | 0.0309 µm |
| Step 8 maximum quaternion distance | 6.77e-5 | 9.38e-8 |
| Step 8 emitted transverse triangle pairs, six pelvic banks | 2,365 | 1,816 |
| Step 8 measured muscle–cortex overlap | 4.250e-6 mL | 2.792e-6 mL |

With variable pleural pressure enabled, a 32-step request accepted 29 consecutive 2 ms steps. On the next post-dynamics left-muscle update the true free-node force residual was 0.015276 N against a 0.0012754 N limit. A separate 29-step launch completed and reproduced exactly the published tissue displacements and surface buffers from before the rejection. Independent FK agreed to 0.0455 µm in relative position and 9.42e-8 in quaternion distance. The completed run measured 0.058 simulated seconds in 14.040 wall seconds including startup, or 0.00413×. The two regional force residuals independently reconstructed at that pose were 0.234 mN and 0.0341 mN. The minimum regional Jacobian across accepted steps was 0.906.

At 58 ms, the exact transverse audit still found 1,771 pairs. The separate closed intersection-volume measurement found 3.306e-6 mL. These six banks do not establish whole-body clearance. Fresh force balance with a finite clearance tolerance cannot establish zero intersecting surface triangles. Whole-body deep-fat loading, full-duration cardiac and respiratory function, zero crossings through motion, sustained throughput and 16 simulated seconds remain open.

The broader atlas included all 2,008 known pairs from the previous 46 ms capture. It included 107,858 anatomically registered face pairs, using clipped opposing finite triangles and keeping both closed-solid queries. Its Metal kernel passed nine independent geometric fixtures. Enabling it with a 50 nm threshold for deepest-point rows rejected after the first corrected 2 ms step. It remains a failed experiment. Candidate coverage alone cannot supply an admissible pressure distribution.

The commands under `authoring` operate on the pinned SSH Numi workspace. The accepted-state files at the failed 32-step run are shared with the byte-identical 29-step replay. Configurations, loaded libraries, authoring identities, independent checks, failed-candidate latches and compressed captures are retained. The anatomy is the existing mixed-source reference assembly.
