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

The native visual check did not produce an image. The installed local renderer
was built from MetalRobo revision `298e5f8565d1cea297c705572493c8d29934d76e`
(binary SHA-256
`9d5aa65222a702d77f1d244525229fb21274fcb9444a20425400be42cf08a3c4`). The
first preregistered command used unsupported focus flags; the second exposed
that this renderer accepts NHBONES1 ABI 2 while the source candidate is ABI 3.
For the third plan, an independently byte-checked diagnostic projection
removed only ABI 3's trailing `source_record_index` from each record. The
renderer then accepted the payload but failed in its native Metal pose pass
before saving a frame. Only the first right-side bone arm was attempted under
each plan; the preregistered stop rule prevented further arms after each
failure. The failure logs, three plans, and result summaries are retained in
the [native-render evidence directory](media/skin-patella-rebase-source-preflight-20261003/).
The ABI 2 payload is a compatibility diagnostic, not a replacement for the
ABI 3 source payload.

The installed renderer supports NHSKIN1 through ABI 4 and cannot consume the
current ABI 5 full-weight candidate. Its ABI 4 top-four diagnostic shell was
not rendered. A compatible current native runtime and a successful Metal pose
pass are required before visual shell review or held-out flexion screening can
close that gap. No board visuals or video were generated.

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
