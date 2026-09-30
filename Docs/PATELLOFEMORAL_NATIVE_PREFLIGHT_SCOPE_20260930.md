# Patellofemoral native contact preflight scope - 30 September 2026

The [exact contact-surface audit](PATELLOFEMORAL_CONTACT_INTERSECTIONS_20260930.md)
found 18 source-authored patella/femur triangle crossings. I reran the Numi
Lab C++ articular-contact preflight against the current `NHKNEE1` left and
mirrored-right payloads, then checked the native outputs with the
[execution receipt](media/patellofemoral-preflight-scope-20260930/receipt.json).
The built probe comes from Lab revision `24f8cdf4`; its contact-model and
probe source files are byte-identical to current `coupled` revision `5787938`.
The receipt binds both source revisions, the binary, both payloads, the exact
intersection receipt, and the complete native stdout.

The CPU preflight passes its own **65-point prescribed 50 µm closure/restore
curve** with bitwise replay and force/energy balance. Its source-rest state is
defined to have zero contact force. The `PTC_To_FMC` row activates all
**11,586** slave nodes over **0.005011 m²** of femoral surface and reports
**1,900.656 N** at peak prescribed closure on the left, with a nearly identical
right result. These values are responses to the artificial closure input,
not forces from the 18 initial surface crossings.

The source names five distinct articular pairs that all reuse the exact same
**22,478 femoral contact faces**. In this preflight, every one of those five
pairs activates the same 11,586 femoral slave nodes. `PTC_To_FMC`,
`TBC-L_To_FMC`, and `TBC-M_To_FMC` also have identical area, foundation
stiffness, pressure, force and energy on each side. The native model's
reference-separation subtraction makes every sampled femoral node respond to
the prescribed closure; it does not localize load to the nearby patellar
contact patch or resolve source initial penetration.

**Localized initial patellofemoral contact admission fails.** The preflight is
useful for checking the bounded elastic-foundation operator and replay, but
its pairwise pressure numbers cannot qualify loaded cartilage contact,
patient anatomy, or a knee force path. The next runtime change needs a
pair-specific current-geometry correspondence, signed physical gap and
initial-penetration policy, followed by an accepted native transaction with
contact work, force balance, rollback and sustained-state checks. No source
faces or compiled payload bytes were altered here.

Reproduce with the retained native build and exact Human payloads. Exit 2 is
the expected failed localized-contact admission after the receipt is written:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m tools.verify_patellofemoral_preflight_scope \
  --binary /Users/home/numi-lab-loaded-knee-numanx-runner-build-20260922/bin/metalrobo_numilab_human_knee_contact_probe \
  --native-repo /Users/home/numi-lab-loaded-knee-numanx-runner-20260921 \
  --current-repo /Users/home/numi-lab-cardiac-native-20260930 \
  --output Docs/media/patellofemoral-preflight-scope-20260930/receipt.json
```
