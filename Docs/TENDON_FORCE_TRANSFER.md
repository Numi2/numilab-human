# Tendon force transfer

## Purpose

The Human already evaluates MyoSim's authored spatial muscle-tendon routes and
applies their tension to the articulated skeleton through the route length
Jacobian. That is the only active rigid-body force authority today. A
BodyParts3D tendon mesh touching a calcaneus is not another force path.

The native enthesis representation is now implemented. Every source route has
one explicit origin and insertion binding. A binding either preserves its
authored bone-owned point or replaces that point with an explicitly admitted
named bone triangle. It is resolved before the authoritative `J^T` scatter and
is never added on top of it.

## Coverage

The production `NHTENDON1` payload contains:

- 416 active muscle-tendon routes;
- 832 mechanical endpoint records;
- 832 unchanged source-site point bindings; and
- zero automatically admitted surface migrations.

Point attachment is already a physical route-to-rigid-body transfer: the
source site belongs to a named articulated bone and contributes its exact
moment arm to the route Jacobian. Triangle mode adds a surface traction
representation; it is not required for the skeleton to receive tendon force.

The first surface-registration candidate covers the bilateral Achilles groups:

- BodyParts3D `FJ1405`/`FJ1405M` tendons and `FJ3360`/`FJ3256` calcanei;
- `gaslat`, `gasmed`, and `soleus` on both sides; and
- Core bodies `calcn_r` 138 and `calcn_l` 152.

The scope is deliberately a native C++/Metal runtime feature. Offline import
may serialize source records, but no Python process may participate in force
evaluation, state update, or rendering.

## Force contract

For a supported source route, let `T` be its already evaluated tensile force,
`a` its terminal attachment, and `F` the corresponding terminal world force.
Let `o` be the distal body's world origin. The enthesis must produce nodal or
triangle traction forces `f_i` at world positions `x_i` such that:

```text
sum(f_i)                  = F
sum((x_i - o) cross f_i)  = (a - o) cross F
```

The terminal point is represented by a named bone triangle plus barycentric
coordinates. If `a` is on that registered triangle, its three barycentric
weights distribute `F` exactly and preserve both equations. A projection that
moves `a` is not silently admitted: it changes the path moment arm and must
either pass a registered-endpoint residual review or become an explicit route
endpoint migration, followed by a new force/moment validation.

The previous BodyParts3D enthesis strip remains visual geometry only. It may
show the same named triangle but never contributes a second spring, weld, or
generalized force.

## Native payload and execution shape

`NHTENDON1` binds every authored route endpoint. Each record stores route,
site, body, attachment mode, resolved local point, and endpoint migration.
Triangle records additionally store the named bone identity, exact source
triangle, local vertices, and barycentric coordinates. Mixed bodies,
incomplete coverage, duplicate endpoints, invalid weights, and unadmitted
surface receipts fail closed.

The native step remains one transaction:

1. Core validates all `2 × muscle_count` endpoint records.
2. A triangle binding receives a route-private resolved site; point bindings
   retain the exact source site.
3. Metal/Core evaluates route tension and `J^T` once from that resolved program.
4. The inspection traction field distributes `f_i = lambda_i F` without
   contributing another generalized force.
5. The native renderer consumes the same resolved site program.

The first release is an attachment-force law, not a deformable tendon
material. A later continuum phase can add a fibre or volumetric tendon state
only after its stiffness, damping, and failure/collision assumptions are
calibrated. BodyParts3D surfaces do not supply those parameters.

## Executable evidence

On the local Apple M4, two consecutive canonical Metal probes were byte
identical. Both reported 416 routes, 832 endpoint bindings, 90 applied wraps,
zero point-program generalized-force difference, and zero point-traction force
and moment residual. CPU/Metal maximum errors were `7.45e-7 m` route length,
`2.63e-3 N` actuator force, `4.72e-3` per-muscle generalized force, and
`6.42e-3` summed generalized force.

The inferred six-insertion Achilles candidate preserved its distributed force
exactly and had maximum moment residual `2.28e-6 N m`, but it required maximum
endpoint migration `0.04990 m`. That changed default route length by up to
`0.01838 m`, actuator force by `180.62 N`, and whole-body generalized force by
`203.26`. Its receipt is therefore marked `mechanical: false`; the production
pack retains the six authored calcaneus point attachments.

The four-angle [right Achilles diagnostic](media/numi-human-right-achilles-mechanics-640/capture.transcript.txt)
was rendered by the native Apple M4 path with `NHTENDON1`, exact BodyParts3D
tendon geometry, named tibia/calcaneus bone geometry, and no synthetic
render-time attachment collar.

This establishes complete route-to-bone mechanics for the active Human and a
working, force/moment-preserving surface-traction mode. It does not claim that
the current BodyParts3D/MyoSim surface registration is accurate enough to
replace the authored Achilles sites, nor does it establish whole-tendon
continuum mechanics, tendon material calibration, gait, injury prediction, or
clinical validity.
