# H706 native launch route preparation

This task-private tool builds a hash-pinned plan for the existing native Human probe. It does not launch the native binary, create a Runtime, or execute Metal/GPU work.

## Current state

`readiness-006.json` reports that the route is waiting for the complete current O/I/D plus 217 volume/registration bundle, candidate geometry admission, conservative mass/density admission, registered-volume importer environment, and current Human scene argv/input pins. It binds the native004 binary/build, all 14 VFS external source contents with per-file SHA-256 and compile-time guard status, the compiled probe source, seven runtime dylib/metallib hashes, and the seven immutable source330 inputs. Runtime binaries and metallibs are hash-bound and are not copied into this task package.

The route pins the full-reference capacity build receipts and overlay-002, the 691 detail bridge source, and immutable source330 parent mass/labels/nodes/source/tets/weights/pose files. Importer limits remain 0.0004 m for parent registration and 0.019353 m for conservative mass transport. The capacity-profile-002 build is source/build/test evidence only; no volume transport or native runtime was executed.

## Import modes

The first probe invocation is source-only and uses `OPERATIONAL_REFERENCE_SPLIT=1` to validate the changed-topology 685 pullback and export the 691 canonical detail-to-boundary map. It sets registered-volume installation and preflight flags, but the probe returns before scene installation or Runtime creation.

After the saved source-only log and canonical map pass validation, the second probe invocation uses `OPERATIONAL_REFERENCE_SPLIT=0`. The current probe only allows split operational-reference pullback during preparation-only source preflight. Physical import uses the normal sourceMassParent route with the same pinned source330 packet and 0.0004 m registration bound. This invocation installs the volume and then runs the requested accepted-state motion; the app has no separate install-only entry point.

Geometry and mass admission receipts are mandatory plan inputs. They must attest the existing positive-cell, closure, shared-I, bone-contact, thickness, protected-149, mass-conservation, density, regional-capacity, and transport-distance gates. The script rechecks their hashes before issuing receipts. No failed or diagnostic geometry is converted into an admission claim.

## Manifest and commands

Start from `route-inputs.template.json`. Fill it with the final current 228/217 bundle, candidate-specific geometry and mass receipt paths and exact claim paths, SHA-256 pins for every input, and the current scene argv/environment. The scene argv must preserve the persistent standing route, include a positive `--muscle-step-count`, end with the single `{RUN_OUTPUT}` placeholder, and pin every absolute file argument. Choose new paths for `output_root` and `detail_map_export_target`; both must be absent.

Use the Numi Python environment:

```sh
PY=/Users/numi/numi-human-prep-venv-20261005/bin/python
TOOL=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-launch-route-preparation-711/prepare_h706_native_route.py
```

Readiness is safe before a complete candidate exists:

```sh
"$PY" "$TOOL" readiness --manifest route-inputs.json --out readiness-006.json
```

When the final inputs and admission receipts exist, create the first-stage plan and command:

```sh
"$PY" "$TOOL" generate --manifest route-inputs.json --out h706-launch-plan-001.json
```

Run the generated source-only command only after the route is explicitly authorized. Save its stdout, stderr, and exit code; then record the source-only result:

```sh
"$PY" "$TOOL" record-source-preflight --plan h706-launch-plan-001.json --stdout source-only.stdout --stderr source-only.stderr --exit-code 0 --out source-only-receipt.json
```

Generate the install-and-motion stage only from that pinned receipt:

```sh
"$PY" "$TOOL" generate --manifest route-inputs.json --source-preflight-receipt source-only-receipt.json --out h706-launch-plan-002.json
```

The second-stage command is also not run by the generator. After authorized execution, `record-motion` requires the registered-volume install PASS marker and terminal audit matching the requested accepted-step count.

## Evidence boundary

The native004 compile/link receipt establishes a prepared binary only. Readiness, generated shell commands, source-only preflight, a surface packet, or mass-map preparation do not establish motion qualification. A final runtime result must be recorded from the actual native log and accepted-state terminal audit. No native runtime or GPU execution was performed while preparing this route.
