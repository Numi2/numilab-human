# Heel support in the registered resting adult

The previous posture left both heels above the bed. We used the measured normal contact Jacobians for 32 bed regions, the existing skin and fat compliance, body weight and joint gravity moments to set the opening posture. The fit uses the spine, its joint-equality followers, both hips and the floating root. A final 0.152 mm lift and 0.151° rotation use the contact points measured in that posture. The six pose-specific shoulder ligament paths were registered again and checked against the 185 cortical banks. The native per-step force kernels and material properties match the control.

| Native measurement | Prior posture | Reseated posture |
|---|---:|---:|
| Right heel normal force | 0 N | 31.2–48.0 N |
| Left heel normal force | 0 N | 46.2–57.8 N |
| Horizontal bed force RMS | 272.0 N | 68.8 N |
| Terminal horizontal center-of-mass speed | 22.4 mm/s | 5.64 mm/s |
| Terminal vertical bed force | 931.3 N | 949.6 N |
| Skin self-intersection pairs | 47 | 28 |
| Skin-to-bone crossing pairs | 204 | 441 |
| Raw skin-to-anatomy crossing pairs, including intended anatomical interfaces | 11,839 | 12,065 |

The Apple M6 run accepted 64 steps of 0.125 ms with variable pleural pressure, 34 ligament force rows, 26 fascia regions and zero root-assistance force or torque. The vertical weight target was approximately 953 N. Initialization still reports a maximum acceleration residual of 103.4 in generalized-coordinate units and dynamic release. The native interval ran at 0.0001754×. The 8 ms process took 61.85 seconds, or 0.0001293× including startup and terminal export. This horizon establishes neither supported rest nor sustained respiration.

The input volume importer reports zero inverted, below-shape and below-volume cells among 1,457,316 tetrahedra. It accounts for 3.45257 kg of skin and 10.48711 kg of fat. All seven native import inputs were rehashed against the archived manifest and the loaded node file matches the surface compiler. This run transfers bed forces through 32 condensed skin/fat regions. The import report describes the source volume. Motion-dependent validity of all its cells is unqualified.

Two experiments failed. Friction-aware recruitment increased horizontal force RMS and terminal horizontal speed by approximately 9%. Larger independent hip rotations caused three ligament paths to fail cortical registration. Both trials and their diagnostics are retained. Applying the smaller, spine-led correction gained real heel support at an anatomical cost. The accepted endpoint has 237 new right pelvic bone crossings and 204 right radial crossings. Reconstructing the uncompressed posed skin from its donor frames reproduces both sets. Transporting the reference skin rigidly with each verified bone finds zero pelvic pairs and 212 radial pairs. This distinguishes a pose-dependent pelvic defect from a registration defect already present at the radius. Terminal fascia and ligament acceptance covers their own force elements.

## Reproduce on SSH Numi

Use the archived registered material, prior native owner and original workspace paths in the recovery manifests. The supported release contains the complete launch argument vector, environment, compilation commands, native executables, captures and inherited file hashes. With the retained Human authoring modules on PYTHONPATH, run the following from the connected-forearms-087 worktree. Use a new output directory for each replay.

```sh
w=recruitment-shear
python common-boundary/run_native.py "$w/reseated-source" "$w/reseated-owner" common-boundary/precise "$w/replay-64" 64
```

The stored experiment can be independently checked with the following command after restoring its evidence. It compares actual per-step heel impulses, accepted time, native pack and binary hashes, variable exterior pressure, root assistance and the cortical and terminal certificates.

```sh
python "$NUMILAB_HUMAN_REPO"/tools/native_ligament_contact/registered_material_volume/regional_bed/supported_pose/verify_supported_release.py recruitment-shear/reseated-64 recruitment-shear/reseated-motion.json recruitment-shear/reseated-crossings.json recruitment-shear/reseated-cortices/registered-fibers.json
```

The verifier and construction sources are in [supported_pose](../../../tools/native_ligament_contact/registered_material_volume/regional_bed/supported_pose). The original sources and native payloads can be recovered from the [supported experiment archive](https://github.com/Numi2/numilab-human/releases/download/human-historical-support-checkpoints-20261122/registered-loaded-bed-20261125-supported.tar.zst) and [rejected shear experiment archive](https://github.com/Numi2/numilab-human/releases/download/human-historical-support-checkpoints-20261122/registered-loaded-bed-20261125-shear.tar.zst). Member hashes were checked against independent downloads.


## Bilateral knee flexion

A second construction preserves the original spine and hip coordinates. It permits a small positive flexion of each knee, uses the source joint-equality derivatives for all followers, and measures gravity and contact moments for the hips, spine and knees. The fit selected 4.89° on the right and 4.85° on the left. The production authoring tool reproduces every argument in both tested native launches. The generated cortical registration now corresponds to the knee-led, weight-calibrated configuration. Both builds remain independently recoverable.

| Apple M6 at 8 ms | Spine-led, calibrated | Knee-led | Knee-led, calibrated |
|---|---:|---:|---:|
| Horizontal bed force RMS | 68.84 N | 55.99 N | 53.33 N |
| Terminal vertical bed force | 949.61 N | 937.49 N | 952.72 N |
| Terminal horizontal center-of-mass speed | 5.64 mm/s | 4.57 mm/s | 4.35 mm/s |
| Right pelvic bone crossings | 237 | 0 | 0 |
| Right radial bone crossings | 204 | 202 | 202 |
| Skin self-intersections | 28 | 39 | 61 |
| Raw skin-to-anatomy pairs | 12,065 | 11,785 | 11,796 |

Both knee-led runs use the same musculoskeletal assets as the control. They carry real bilateral heel loads on all 64 accepted steps. The calibrated configuration reached 0.0001766× over steps 1–64 and 0.0001304× including startup and export. Its maximum generalized acceleration residual remains 117.5. The density ledger contains 96.65624 kg of articulated mass and 0.47571 kg of cooked deformable tissue, totalling 97.13195 kg.

All 61 calibrated endpoint fold witnesses reproduce on the emitted skin. The same 61 face pairs yield zero intersections at source coordinates and seven under an independent uncompressed posed-skin reconstruction. The maximum difference in the affected native and reconstructed vertex positions is 0.000110 mm. This check describes those particular witnesses. Submicron sensitivity must be fixed and cannot exempt a failing triangle pair.

A requested 256-step extension reproduced the first 64 articulated position and velocity fingerprints and all checked physiological columns exactly. It accepted 98 steps (12.25 ms). At that point the bed carried 954.57 N, both heels still reacted, and root assistance remained zero. Step 99 exhausted 1,024 active Krylov columns and failed the fresh mechanical momentum certificate. Its peak row was 1176, in the left superior internal-oblique attachment sheet. The vascular residual was below its threshold. The common-field and respiratory failure record retained the previous accepted state. A separate trial with coupled row equilibration failed after nine accepted steps. The calibrated unscaled configuration remains the reference for this failure.

Use `--knee-flexion` with the existing pose fitter. The source, commands and native payloads for both 64-step knee trials are retained in the [knee support recovery archive](https://github.com/Numi2/numilab-human/releases/download/human-historical-support-checkpoints-20261122/registered-loaded-bed-20261125-knee-support.tar.zst). The exact endpoint and force measurements are in [knee-flexion](knee-flexion/).
