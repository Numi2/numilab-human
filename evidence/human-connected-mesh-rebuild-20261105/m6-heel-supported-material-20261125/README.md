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

The Apple M6 run accepted 64 steps of 0.125 ms with variable pleural pressure, 34 ligament force rows, 26 fascia regions and zero root-assistance force or torque. The vertical weight target was approximately 953 N. Initialization still reports a maximum acceleration residual of 103.4 m/s² and dynamic release. The native interval ran at 0.0001754×. The 8 ms process took 61.85 seconds, or 0.0001293× including startup and terminal export. This horizon establishes neither supported rest nor sustained respiration.

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
