# Native-current skin registration sidecars

This packet records a strict initialization/reference registration of the conditioned outer skin and paired inner/fat surfaces against the actual native run006 observation and capture016 parent FEM. It does not admit a changed skin/fat volume, mass transfer, or physical motion.

## Result

The sidecars contain 32,002 O nodes, 15,022 I nodes, and 15,022 D nodes (62,046 typed node rows total), with 63,959 O faces. Source topology and source ancestry are retained separately from current-parent registration. The current-parent registration was checked against capture016 current coordinates and labels by the native C++ checker included below.

The native check passed all 62,046 rows:

- maximum parent reconstruction error: O `2.22478e-16 m`, I `1.38778e-17 m`, D `2.77556e-17 m`;
- same-material, same-geometric-donor, same-anatomical-region checks passed for explicitly inferred rows;
- wrong-donor and wrong-region negative regressions passed;
- `physical_steps=0`; `changed_volume_admitted=false`.

The exact command and stdout are pinned in `native-current-parent-check.receipt.json` and `native-current-parent-check.stdout.txt`. The checked-in source is `native-current-parent-check.cpp`; the receipt also pins the executable hash and original task path.

## Six constrained current-parent mappings

Six O rows had an old accepted-transfer parent registration farther than the 2 cm gate. Their exact D37 detail-source coordinates are also 22.2-36.3 mm away from the current candidate O point. The current sidecar keeps those source identities intact and records separate, explicit `inferred_current_parent_tet_same_material_donor_region` registrations. Selection is constrained to material 15 and the same source geometric donor and anatomical region; distance is minimized first, then the lowest tet row resolves indistinguishable ties. The maximum chosen parent distance is 8.204 micrometers.

| O row | donor / region | old accepted parent distance | exact D37 source distance | current parent distance | parent tet / stable face |
|---:|---:|---:|---:|---:|---:|
| 13498 | 42 / 42 | 22.101 mm | 22.384 mm | 7.280 micrometers | 4031 / 14318 |
| 13499 | 42 / 42 | 21.905 mm | 22.227 mm | 8.203 micrometers | 4031 / 14318 |
| 16946 | 150 / 150 | 22.099 mm | 22.207 mm | 0.0527 micrometers | 8707 / 63821 |
| 17203 | 150 / 150 | 36.305 mm | 36.307 mm | 0.0600 micrometers | 9168 / 70119 |
| 17204 | 150 / 150 | 27.068 mm | 27.067 mm | 0.0855 micrometers | 8905 / 66584 |
| 17223 | 150 / 150 | 28.274 mm | 28.328 mm | 0.126 micrometers | 9168 / 70119 |

This is inferred mechanical reference registration, not source or geometric ancestry. The 62,046-row receipt preserves the complete node mappings; large CSVs are intentionally not duplicated in this Human evidence folder.

The kind-0 provenance case is kept distinct: O033 row 16398 maps to compact O row 16264 and baseline806 coarse source row 1233. P1 detail row 24939 is binding-weight support only. The selected current parent is capture016 tet 13455, stable face 98586; it is not D37 source identity.

## Inputs and hashes

`source-pins.json` records canonical source payload and 86-weight table hashes, conditioned O and compact-row-map hashes, run006 native O/pose hashes, capture016 parent cell/node/tet/label hashes, attempt018 output hashes, and strict-check hashes. `attempt-018-receipt.json` retains the complete input/output hash map and method details.

The canonical ABI5 payload SHA-256 is `7a6c44b641b286497fbef73157e2945e2b7c4d5d2c423eff3b26590aa496abf4`. The full 86-weight table SHA-256 is `85c0a612b48279d88c8f00b9e3eea46e480f9ee243df71093c82fb0d9556d98f`. The conditioned O SHA-256 is `5e686b14d9360267f4b40bbeb5ed6c9c68bf871933874b5aa8eaa5898757ec2b`; the compact O map SHA-256 is `5ed0c63d4a35a023bd91d0cbec1a933c8ad1f7b5fc1d32148cd8568ac77db8f4`.

## Source and commands

The exact exporter and six-row diagnostic scripts are copied here. Their pinned task inputs live under `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/`; attempt018 remains immutable at `volume-worker/registered-surface-sidecar-222/attempt-018-native-current-reference`.

The scripts use fixed task paths. The exporter refuses to overwrite an existing attempt018 output directory. For future replay, copy the exporter to a fresh task-local scratch location and retarget only its output directory, preserving all pinned input hashes. Do not rerun the diagnostic in place because it writes its report into the attempt016 source directory.

```sh
/Users/numi/numi-human-prep-venv-20261005/bin/python /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/registered-surface-sidecar-222/diagnose_six_parent_registration_227.py
/Users/numi/numi-human-prep-venv-20261005/bin/python /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/registered-surface-sidecar-222/export_native_current_sidecars_228.py
/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/P/native-current-parent-registration-229/check /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/registered-surface-sidecar-222/attempt-018-native-current-reference /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv --expect-pass
```

The first two are the source-preparation/export commands; their retained outputs and hashes are authoritative here. The final command is the exact native check recorded in its receipt and exited 0.

## Limits

This proves strict current-parent reference registration and reconstruction only. It does not establish geometric nesting, a positive changed child tetrahedral volume, conservative mass transfer, accepted physical motion, or skin/fat qualification. A separate run006 exact self-audit retained 78 nonadjacent rational skin intersection pairs; this packet does not resolve them. Prior failures remain at their original task paths.
