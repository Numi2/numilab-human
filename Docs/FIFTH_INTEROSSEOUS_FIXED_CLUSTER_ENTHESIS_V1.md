# Bilateral fifth palmar-interosseous origin ownership

The bilateral `UI_UB5` origin endpoints now transfer their terminal load to
distributed envelopes on the named BodyParts3D fifth metacarpals. The MyoSim
route points, route bodies, path geometry, force parameters, and source-route
`J^T` authority remain unchanged.

## Source and anatomical identity

The pinned source is `MyoHub/myo_sim` revision
`33c89c2bde282553dde3f526768eb3bdcfaa7649`. Its model documentation defines
`UI-UB` as the palmar or ulnar interosseous. In the exact fifth-ray route,
`UI_UB5-P1` is stored on the third-metacarpal kinematic carrier and the next
site, `UI_UB5-P2`, is stored on the fifth metacarpal.

Default-pose common-frame surface comparison gives:

| Side | Fifth metacarpal | Fourth metacarpal | Third metacarpal |
| --- | ---: | ---: | ---: |
| right | `1.04854 mm` | `8.8048 mm` | `15.3678 mm` |
| left | `2.25093 mm` | `9.0707 mm` | `14.7681 mm` |

The fifth metacarpal is therefore both the nearest exact registered surface
and the source/anatomical fifth-ray owner. The compiler does not choose the
also-near fourth metacarpal by distance alone.

Relevant sources are the
[pinned MyoSim repository](https://github.com/MyoHub/myo_sim/tree/33c89c2bde282553dde3f526768eb3bdcfaa7649),
[official MyoSuite muscle glossary](https://myosuite.readthedocs.io/en/stable/suite.html),
and a recent fifth-ray enthesis study identifying the third palmar
interosseous as the fifth-ray adductor
([Karakostis et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873511/)).

## Fixed-cluster mechanics

The exact `NHRIGID` tree proves that the third and fifth metacarpal bodies are
zero-DoF fixed siblings of the same capitate body on each side. Their relative
transform is invariant. The compiler therefore expresses the fifth-metacarpal
surface nodes in the unchanged third-metacarpal route-site frame and then runs
the ordinary 12 mm distance, 12 mm radius, force/moment conservation, and
amplification gates.

This is not a route reparenting or endpoint migration. It changes no muscle
path length or moment arm. Compilation fails if either metacarpal gains a
relative DoF, if the fixed parent changes, or if the rigid payload drifts.

The admitted envelopes are:

| Side | BodyParts3D member | distance | radius | force amplification |
| --- | --- | ---: | ---: | ---: |
| right | `FJ3358` | `1.04854 mm` | `5.04414 mm` | `1.13017` |
| left | `FJ3252` | `2.25093 mm` | `10.8033 mm` | `1.22037` |

Both endpoint migrations are zero. Compiled force residuals are below
`2.25e-16`; compiled moment residuals are below `1.01e-18 m`.

## Apple Metal qualification

The resulting `NHTENDON3` payload contains all 832 endpoint laws: 641
distributed envelopes and 191 explicit point fallbacks. Of the envelopes, 631
are registered BodyParts3D bone surfaces.

On Apple M4 Pro with Metal API validation enabled, the reference probe
executed all 832 transfers. Maximum force residual was `0.000246159 N`, maximum
moment residual was `7.17526e-6 N m`, maximum nodal CPU/Metal parity error was
`0.000129922 N`, and tendon replay was byte-identical.

A separate two-step `0.1 ms` persistent transaction applied a `0.2` activation
increment to both `UI_UB5` routes (source muscles 272 and 335). It executed
1,664 endpoint transfers, including 1,282 envelope transfers. The borrowed
consumer used the exact same command-buffer snapshot, direct rigid-state effect
remained bitwise absent, injected rejection rolled back, and replay was
bitwise. Source-route `J^T` remained the sole rigid-force authority.

The machine-readable receipt is
[`m4-pro.json`](media/numi-human-fifth-interosseous-fixed-cluster-v1/m4-pro.json).

## Evidence boundary

This resolves two anatomically misclassified point fallbacks. It is not a
deformable interosseous tendon, subject-specific enthesis footprint, pulley or
extensor-hood material validation, independent fifth-CMC mechanics, static
equilibrium certificate, or clinical model. If future mechanics introduce
relative metacarpal motion, this fixed-cluster representation must fail and be
replaced by an explicitly articulated attachment owner.
