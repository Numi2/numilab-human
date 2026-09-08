# Current-pose foot support

Native commit `329d54b9e0b303bb377c04e2a2eef7272d3ba4f1` evaluates the complete
source foot primitives against the ground at each pose. Eight capsules compile
to sixteen endpoint-sphere constraints; the two heel ellipsoids each retain
an exact support query. The independent MuJoCo 3.12 check passes for the
compiled initial pose. Internal equilibrium and sustained motion remain open.

NHCNT2 stores source identity, COM-relative geometry, the authored plane and
friction. One native decoder serves the visual/standing probe and NumanX v5.
C++ performs offline placement/recruitment; Metal evaluates contact and dynamics.
The point Jacobian uses the velocity of the surface material, including the
frictional moment arm, rather than the tangential derivative of the moving
closest-point location. Matter receives matching geometry and candidate
Jacobians. Original NHCNT1 point witnesses retain their semantics.

The shared articulated point-query record is now 96 bytes, and Matter's Human
support contact record is 80 bytes. Its point-query mirror has the same layout;
the runtime ABI fingerprint includes the changed point type. Rebuild native
consumers and metallibs together. Contact capacity is 32 bounded rows, with
18 rows used by this source; muscle capacity is unchanged.

| Check | Result | Boundary |
|---|---|---|
| Source geometry at FP32 initialization precision | Minimum full-primitive gap +1.5912e-8 m; maximum conversion error 3.4100e-9 m | 57 source Python/XML/config files checked against the archive |
| Independent gravity wrench | Maximum force residual 2.8387e-5 N; moment residual 1.8093e-5 Nm | Final recruited pose, six loaded contacts, tolerance 1e-3 |
| Native support wrench | 952.864475213 N against weight 952.864477038 N; root force residual 1.8252e-6 | Static unilateral support; internal normalized RMS 12.3142, `internal_balanced=false` |
| Generic Metal surface query | Sphere/ellipsoid point, Jacobian, impulse and inverse-mass parity pass; malformed sphere rejected transactionally | Analytic fixtures, existing G1 regressions retained |
| Matter surface contact | Point/sphere/ellipsoid restoring-force derivative error 1.90735e-5; friction moment arm and exact rollback pass | Isolated owning contact kernels, not full anatomical v5 acceptance |
| Human one step | 18 contacts; minimum native gap -9.3132e-9 m; no root assistance | 0.1 ms, `balanced=false` |
| Human 64 steps | Bitwise replay; maximum generalized velocity change 0.929417; worst native gap -9.0882e-5 m | 6.4 ms diagnostic; generalized maxima mix coordinate units |
| Independent final-state check | Full primitive gap -8.6597e-5 m; left knee exceeds its lower limit by 0.00340006 rad | **Dynamic contact and joint limits fail** |
| Regression checks | 191 Python tests, seven existing skips; native CTest 2/2; both focused GPU probes pass | M4 Pro; concurrent NumiVivo workload excludes performance qualification |

The [receipt](media/curved-support-20260908/receipt.json) retains all final checks
and intermediate failures. The first GPU probe exposed an omitted function-table
binding in that probe; it is repaired. The first independent wrench comparison
used the placement seed with forces from the subsequently recruited pose. The
probe now exports its final `compiled_equilibrium_q`, and the corrected check
passes. Neither rejected check was overwritten. The 6.4 ms source check still
fails and remains visible.

## Reproduce

Use the pinned source environment with MuJoCo 3.12, then author contact geometry
without rerunning the existing muscle-architecture fits:

```sh
PYTHONPATH=src python -m numilab_human.myosim_export --sources Sources --output source-export.json
PYTHONPATH=src python -m numilab_human.support_primitives --source-export source-export.json \
  --source-manifest Build/myosim-fullbody-with-equalities/myosim-fullbody-reference.manifest.json \
  --output Build/curved-support
```

The incremental command verifies the source binding and referenced rigid bytes.
It creates a new output directory. A normal `myosim-build` also emits NHCNT2.
Place the other pinned profile payloads beside it, then run on the native host:

```sh
PYTHONPATH=src python3 -m numilab_human.cli support-stance \
  --profile config/myosim-support-stance.v2.json \
  --input INPUT --runtime-build NATIVE_BUILD --output NEW_RECEIPT
```

The existing v1 profile remains available. Verify the published bounded evidence:

```sh
python3 Docs/media/curved-support-20260908/verify_curved_support.py
```

## Remaining completion work

Complete internal muscle/fibre equilibrium and consistent joint-limit/contact
reactions, then admit the identical prepared pose into registered NumanX v5
tissue. The current offline limit reactions and legacy sequential contact path
do not establish joint-limit complementarity. Implement source-compliant dynamic
limits and accepted-root telemetry before qualifying the Brain standing/recovery
controller and the frozen 420 standing/recovery/walking trials.

The costal callback timeout remains unresolved. No long costal run was launched
while the separate NumiVivo cohort occupied the Mac mini. The earlier timeout
receipts remain unchanged; contention has not been established as their cause.
Material calibration, loaded anatomical tissue and full-release qualification
also remain open.
