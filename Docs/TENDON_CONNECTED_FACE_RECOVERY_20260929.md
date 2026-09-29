# Connected-face tendon surface recovery — 29 September 2026

The two attachment regressions exposed by the bounded bilateral knee repair are
closed: `vasint_l` insertion and `vasmed_l` origin again have distributed
attachments on their named tibial and femoral surfaces. Six other source-point
fallbacks also gain surface admission. This increment leaves bone geometry,
source sites, joint laws, force laws and admission limits unchanged.

## Cause and repair

A coarse triangle can contain useful surface points within the 12 mm patch
without any mesh vertex lying inside that patch. The previous vertex compass
and seed-triangle fallback could miss a well-conditioned force stencil.
The new fallback searches shared-edge faces through bounded segments inside
the original triangles. Every selected point retains its original triangle,
barycentric coordinates and an explicit surface path no longer than 12 mm.

The search is conservative, with one retained entry per visited face. It does
not compute globally shortest geodesics or exhaust all possible surface points.
Four-point quadrature is sampled from deterministic extrema in 26 source-frame
support directions. A passing stencil still must meet the existing force,
moment, distance and amplification gates. Disconnected fragments cannot be
borrowed by this fallback, and no anatomy is warped to fit an endpoint.

| Result | Before this increment | After |
|---|---:|---:|
| Mechanical endpoints | 832 | 832 |
| Distributed attachments | 642 | 650 |
| Registered bone attachments | 632 | 640 |
| External surfaces without native bone verification | 10 | 10 |
| Source-point fallbacks | 190 | 182 |
| Explicit named endpoint migrations | 18 | 18 |
| Previously admitted endpoints downgraded | — | 0 |

All 832 source endpoint laws **and resolved points** are identical to the paired
knee-repair payload. The 18 prior named migration positions are unchanged. One
previously admitted `TRImed_l` origin changes from bilateral projection recovery
to a direct patch on its named target surface. Its quadrature changes; this is
not a claim that every existing envelope byte is preserved.

The eight newly admitted endpoints are `glmax3_r` insertion, `addmagDist_l`
insertion, `fhl_l` origin, `glmax2_l` insertion, `glmax3_l` insertion, `vasint_l`
origin, `vasint_l` insertion and `vasmed_l` origin. The two repaired regressions
have sampled force amplification **1.5295** and **1.6325**, below the unchanged
limit **4**. The ordinary source-to-surface distance and patch radius remain
12 mm; the existing explicit migration limit remains 25 mm.

## Executed native and serialized-payload checks

Append-only local proof root: `Build/tendon-patch-recovery-20260929`.

- Full 832-endpoint compile: exit 0, 51.338 seconds; NHTENDON3 ABI 3 unchanged.
- Connected-face/source regressions: 22 passed, 92 unrelated tests deselected.
  The actual repaired knee inputs demonstrate failure with the new fallback
  disabled and admission with it enabled, without changing their source point.
- Native binding suite: **17 passed**, no skips, 70.67 seconds. It executes
  matched geometry and rejects stale muscle/bone identities, nonexistent
  triangles, wrong owners, off-surface force-conserving nodes, and incorrectly
  displaced named migration points before calibration or dynamics.
- Independent full-payload check: all **9** face-path envelopes / **36** nodes
  reconstruct from the consumed original triangles. Shared-edge continuity,
  per-segment triangle containment and summed 12 mm path limits pass. Maximum
  serialized unit-force residual is `3.095e-8`; maximum corresponding moment
  residual is `2.482e-9 m`.
- Native reference probe on **Apple M4**: exit 0; all **832** endpoint transfers
  and **650** envelope transfers execute through the paired high/low geometry
  path, with byte-identical tendon replay. Maximum GPU nodal parity error is
  `1.564e-4 N`, force residual `1.221e-4 N`, moment residual `5.967e-6 N m`.
  This reference probe does not load NHBONES and reports
  `tendon_geometry_identity_verified=false`; the separate visual/native binding
  suite verifies the matched bone surfaces.
- Same-bone before/after mechanics diagnostics: both exit 0 after 16 native
  1 ms steps with no root assistance. The new payload executes 13,312 endpoint
  transfers, including 10,400 envelope transfers and 2,912 point transfers.
  The exact borrowed consumer and rejection rollback checks pass. Each payload
  independently preserves the source-JT rigid state bitwise; the new payload
  also passes requested deterministic replay. Maximum force and moment
  residuals are `6.115e-5 N` and `8.983e-7 N m` in both runs. This 16 ms check
  does not establish sustained standing or loaded articular qualification.

| Artifact | SHA-256 |
|---|---|
| New tendon payload | `331b210f4606bb0ae066bc2b6f378c3e9bbbdcbf41e108307c2e2acd5607e431` |
| Unchanged corrected bones | `2aaf0567e6a5131c88b599cd56cb605b9d585c2792084b324e84699cabbc9b33` |
| New thin reference client | `fa2bb1a59bf7015a674dc469c0e887b796874b1aa08911e97225e427a333c43b` |
| Reference client source | `c6230b6fb5666267384010704178560cbc655b8e1d05267b5b7920075611afb4` |
| Unchanged native runtime | `6b9062e84bdacd6b86b413d5c549cd0046d9a271fb0caa141e03db41ffa213f6` |

The reference client was built from the published native source and linked
against the retained runtime without modifying that runtime or the separate
dirty sparse-operator worktree. Exact commands, environments, source snapshots,
terminal logs and payload comparisons remain in the proof root.
The final Python source differs from the executed compiler/test snapshot only
in an explanatory comment; independent AST comparison confirms identical
executable source semantics.

## Remaining qualification boundaries

The [bilateral knee geometry repair](BILATERAL_KNEE_GEOMETRY_REPAIR_20260929.md)
still passes its 320 posed interfaces and 160 bilateral checks. Five authored
source-range conflicts remain; this increment does not modify them. The 182
remaining point dispositions comprise 152 distance failures, 24 bodies without a
registered bone surface, four semantic spread failures and two source-model
non-bone endpoints. Their gates are not relaxed.

Ten external surfaces still lack native bone-surface verification. Surface
paths and numerical force transfer do not establish subject-specific entheses,
cartilage-facing orientation, loaded articular contact, tissue material
calibration, clinical anatomy or whole-body control qualification. The retained
10-second September 28 standing video predates these anatomy repairs.
