# NumiLab Human tendon attachment v2

> `NHTENDON2` remains the default source-point-preserving program documented
> here. The qualified, opt-in `NHTENDON3` foot mode adds 18 route-private exact
> named-bone terminals while keeping all bone transforms fixed; see
> [fixed-bone foot entheses v1](FIXED_BONE_FOOT_ENTHESES_V1.md).

## Outcome

`NHTENDON2` adds a mechanically executed tendon-to-bone surface transfer law
without changing any authored MyoSim/OpenSim endpoint. The exact source muscle
route still determines path length, wrapping, activation-dependent force, and
the terminal force direction. An admitted attachment distributes that force to
four source-registered BodyParts3D bone-surface nodes while preserving the
source-point resultant force and moment. Normally those vertices belong to one
named member. The only multi-member exception is the explicit EDL/FDL semantic
map across the four named lesser-toe distal phalanges.

The production compile currently covers all 832 origin/insertion endpoints:

- 304 simulation-inferred, distributed BodyParts3D bone-surface envelopes;
- 528 explicit MyoSim source-site point laws;
- zero endpoint migration;
- zero direct joint-torque records.

Point laws are an intentional mechanical fallback, not missing records. They
retain the exact force transfer already provided by the route Jacobian until a
surface registration is defensible.

## Why this is the pragmatic free-data boundary

BodyParts3D 4.0 remains the geometry authority and OpenSim/MyoSim remains the
biomechanics authority. BodyParts3D publishes a skeletal-muscle
origin/insertion terminology workbook, but it describes attachment anatomy in
text rather than supplying geometric enthesis coordinates. Its mesh archive
also contains only limited named tendon surfaces. No superior freely available
whole-body enthesis-coordinate atlas was found that can replace the selected
stack without a new cross-source registration problem.

The compiler therefore admits an inferred surface only when all of these are
true:

1. the endpoint body owns exactly one registered `NHBONES1` member, except for
   the named lesser-toe EDL/FDL maps described below;
2. the exact endpoint-to-triangle distance is at most 12 mm;
3. four nodes are reachable on one connected mesh patch within 12 mm;
4. the precomputed distribution conserves unit force within `2e-6` and unit
   source-point moment within `2e-8 m` in FP64;
5. the sampled sum of nodal force magnitudes is no more than 4 times the
   terminal force.

Bodies with several bones, absent bone geometry, distant surfaces, or an
ill-conditioned patch fail closed to a source-site point law. The only
multi-member records admitted are eight bilateral EDL/FDL insertions whose four
lesser-toe distal phalanges are enumerated in
[the toe-enthesis receipt](TOE_ENTHESIS_V5.md). The current rejection counts are
190 unmapped multi-member bodies, 24 bodies without a registered bone surface,
236 distance failures, 50 conditioning failures, and 28 patches with fewer
than four reachable vertices.

## Force-transfer law

For source point `a`, attachment nodes `x_i`, and terminal force `F`, the
offline compiler stores four matrices `M_i` such that

```text
f_i = M_i F
sum(f_i) = F
sum((x_i - a) cross f_i) = 0
```

The maps are the minimum-L2 solution of the six force/moment constraints. The
moment equations are scaled by patch radius during factorization for numerical
conditioning; the represented physical moment is unchanged.

The Apple Metal pass consumes the exact wrapped endpoint gradients emitted by
the owning MyoSim route kernel. It rotates the terminal force into the source
body frame, evaluates all four nodal forces, rotates them back to world space,
and projects them through the same articulated body spatial Jacobian probes.
It reports the difference between the distributed `J^T f_i` and the original
source-point `J^T F`; it never writes an invented joint torque.

## Binary and provenance contract

`NHTENDON2` binds three immutable inputs:

- MyoSim source archive SHA-256;
- `NHMYO1` muscle payload SHA-256;
- exact `NHBONES1` payload SHA-256 plus its registration fingerprint.

The payload contains 832 endpoint records and one 288-byte envelope record for
each admitted endpoint. Every endpoint record retains its exact source local
point. The native decoder accepts legacy `NHTENDON1`, but the v2 Metal packer
rejects triangle-migrated programs and accepts only source points or distributed
envelopes. The owning Numi Lab runtime code revision used for the current
qualification is `45fede450ba889b8feb1df0a8330db3c31706497` on `coupled`.

## Reproduce

```bash
numi human numi-human-tendon-envelope-payload \
  --artifact Build/myosim-fullbody \
  --bone-artifact Build/bodyparts3d-myosim-major-bones-v4 \
  --output Build/numi-human-tendon-v5

/path/to/metalrobo_numilab_human_myosim_reference_probe \
  Build/myosim-fullbody/myosim-fullbody-core-reference.nhrigid \
  Build/myosim-fullbody/myosim-fullbody-muscle-reference.nhmyo \
  Build/numi-human-tendon-v5/numi-human-tendon-attachments.nhtendon \
  --metal
```

The qualified probe evaluates all 416 routes and 832 endpoints. The Mac mini
Apple M4 Pro result transferred all 832 endpoints, including 304 envelopes,
with maximum Metal residuals of `1.25827195006e-4 N`,
`2.82619680547e-6 N m`, and `6.103515625e-4` in the generalized-force
correction. CPU/Metal nodal-force disagreement was `8.82795095549e-5 N`, and
the in-process Metal replay was byte-identical. The retained
[current qualification](TOE_ENTHESIS_V5.md) and historical
[v2 reference transcripts](media/numi-human-tendon-attachment-v2-2048/reference/)
make the device, input generations, and counters inspectable.

## Four-angle anatomy inspection

The reviewed 2048 px evidence uses two routes for which both endpoints pass
the v2 gates:

- right anconeus (`ANC`, source actuator 228) between bodies 41 and 42;
- right subscapularis (`SUBSC`, source actuator 215) between bodies 41 and 34.

Each capture retains only the matching exact BodyParts3D muscle surface and
the two endpoint bone owners. The selected actuator receives `0.2` excitation
for one 100 µs step; Apple Metal still evaluates all 416 source routes before
the bounded Core FP64 update and final Metal pose. The cyan geometry is the
unchanged source route. The warm terminal fans and connected footprints use
the exact four nodes loaded from `NHTENDON2`; no render-time nearest-point
projection or collar is used.

The anconeus views retain 182 / 446 / 940 / 1,160 envelope pixels from front
through rear. The subscapularis views retain 443 / 287 / 106 / 1,463. Visual
inspection confirmed that the source route and envelope terminate at the
named bone surfaces in all eight images, including the lateral and posterior
views that expose the footprint best. The [anconeus record](media/numi-human-tendon-attachment-v2-2048/anconeus/capture.transcript.txt),
[subscapularis record](media/numi-human-tendon-attachment-v2-2048/subscapularis/capture.transcript.txt),
and [checksums](media/numi-human-tendon-attachment-v2-2048/checksums.sha256)
retain exact device, image, and execution evidence.

The images are exposed mechanical-anatomy diagnostics. They improve connection
quality and legibility but are not a photorealistic exterior, skin/fat/fascia
model, or proof of tissue-level stress distribution.

## Evidence boundary

This is a live force-transfer law and exact articulated-Jacobian execution, not
a cosmetic tendon line. The admitted surface coordinates are nevertheless
simulation-inferred from a cross-source registration. They are not
source-authored enthesis measurements, a clinical attachment certificate, a
deformable tendon continuum, calibrated tendon damage mechanics, or validation
of anatomical stress distribution. The persistent stand now recomputes,
validates, and publishes those terminal and four-node loads in every accepted
step. It also exposes a borrowed same-command-buffer consumer boundary whose
exact snapshot and rollback were qualified on Apple M4 Pro. A production
deformable tendon/bone solver has not yet assembled these loads into material
state, so tissue stress and deformation remain open. The explicit point
fallbacks must remain until better attachment data or an endpoint-specific
registration receipt is available.
