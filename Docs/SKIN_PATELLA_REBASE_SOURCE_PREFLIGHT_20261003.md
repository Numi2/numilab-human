# Patella-rebased skin source preflight — 2026-10-03

The updated MyoSim source places both patellae anterior to their pinned source
knee-anchor planes. The source rebase audit preserved eight source-equality-
projected pose sets with a maximum body-transform difference of
1.15×10⁻¹⁶ m, and measured at least 33.59 mm of patella anteriority at literal
qpos0. This is source-model geometry evidence, not clinical tracking or loaded
knee validation. See the [patella rebase receipt](media/myosim-patella-qpos0-rebase-20261003/receipt.json).

Using the matching current rigid artifact and lower-limb registration, the
team regenerated the BodyParts3D FJ2810 outer-shell visual payload. An
independent CPU audit rechecked the source archive and mesh identity, all 86
runtime influences for each of 54,949 vertices, all 109,183 triangles, exact
coincident-seam weights, binding frames and rest reconstruction. It passed
with a maximum double-precision full-weight partition error of 5.4×10⁻¹⁴,
float32 runtime partition error of 2.98×10⁻⁷, zero seam-weight mismatch and a
maximum source rest reconstruction error of 4.54×10⁻¹⁴ m. The independent
source-surface harmonic equation residual was 1.11×10⁻¹⁵.

The [receipt and registration](media/skin-patella-rebase-source-preflight-20261003/receipt.json)
and [payload manifest](media/skin-patella-rebase-source-preflight-20261003/payload-manifest.json)
bind this source-only result. The 57 MiB payload and binding solution are
retained under ignored `Build/skin-patella-rebase-source-preflight-20261003/`.
They are visual inputs, not collision or physics geometry.

This does not close the skin gap. The regenerated payload has not been
rendered in the native runtime or screened in the held-out 1.2-radian knee
pose. The earlier 5% geodesic-weight trial still has a documented held-out
pair-level regression; this new payload is a separate source-registration
candidate and has no high-flex result yet. Dynamic skin embeddedness, the
clinical validity of inferred weights, skin thickness/material/contact and
force-coupled soft tissue remain open. The exact historical source binding
payload and registration could not be recovered, so this is a new candidate,
not a reproduction of that missing run.

## Native visual status

The bilateral source rebase does resolve the specific default-pose front/back
defect: all 72 vertices on each compiled patella are anterior to the pinned
knee-anchor plane at literal source `qpos0`, by at least 33.589 mm on the left
and 33.591 mm on the right. Eight equality-projected pose sets retain the
prior body transforms within `1.15e-16 m`. This confirms source geometry
orientation in the sampled states; it does not certify clinical landmarks or
loaded patellofemoral tracking.

The initial native attempts exposed two compatibility gaps: the installed
probe binary did not accept the focus flags present in its checked-out source,
and that runtime accepts NHBONES1 ABI 2 while the current source candidate is
ABI 3. A third attempt with an independently byte-checked ABI 2 diagnostic
projection failed in the Metal pose pass without useful status detail. The
projection strips only ABI 3's trailing `source_record_index`; all body, pose,
geometry, normal and triangle data stay byte-identical. It is a compatibility
diagnostic, not a replacement for the ABI 3 source payload.

I rebuilt the probe object from the current Objective-C++ source using the
project's strict warning flags and linked it against the existing MetalRobo
libraries. The complete CMake target remains blocked by an unrelated
`NeuronCulture.metal` unused-parameter warning treated as an error. With the
updated probe, both one-pass side views completed on Apple M4 at literal source
`qpos0`: the right and left frames each contain the requested patella, femur
and tibia, with 15,057 and 16,131 bone pixels respectively. In both views the
patella is on the anterior side, matching the independent source-plane check.
These are bone-only frames: native logs report zero cartilage, meniscus,
ligament and tendon pixels, so the images do not judge patellar cartilage
contact or soft-tissue fill. The [right frame](../Build/skin-patella-rebase-source-preflight-20261003/native-render-v4-right-bone_anatomy/myosim-fullbody-articulated-bodyparts-bones-focus-body-142-side.png)
and [left frame](../Build/skin-patella-rebase-source-preflight-20261003/native-render-v5-left-bone_anatomy/myosim-fullbody-articulated-bodyparts-bones-focus-body-156-side.png)
are retained locally under ignored `Build/` outputs.

The installed renderer supports NHSKIN1 through ABI 4 and cannot consume the
current ABI 5 full-weight candidate. The ABI 4 top-four diagnostic projection
did render successfully as a side silhouette with 18,454 skin-shell pixels.
That confirms only static visual payload consumption; it is not the ABI 5
candidate and does not test deformation or high-flex embeddedness. The
renderer plans, results, output hashes and source-only receipt are retained in
the [native-render evidence directory](media/skin-patella-rebase-source-preflight-20261003/).
No board deck or video was generated.

Re-run the source-only gate against the retained local artifact and payload:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m numilab_human.skin_source_payload_preflight \
  --sources Sources \
  --artifact Build/myosim-fullbody-patella-rebase-core-audit-20261003 \
  --registration Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json \
  --payload Build/skin-patella-rebase-source-preflight-20261003/payload/bodyparts3d-myosim-skinned-shell.nhskin \
  --output Docs/media/skin-patella-rebase-source-preflight-20261003/receipt.json
```
