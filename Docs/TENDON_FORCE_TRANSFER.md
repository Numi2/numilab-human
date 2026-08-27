# Tendon force transfer

## Purpose

The Human already evaluates MyoSim's authored spatial muscle-tendon routes and
applies their tension to the articulated skeleton through the route length
Jacobian. That is the only active rigid-body force authority today. A
BodyParts3D tendon mesh touching a calcaneus is not another force path.

The next physical milestone is a native enthesis representation: a source
tendon endpoint attaches to named triangles of its registered bone and
distributes the existing route tension as a traction field with the same
resultant force and moment. It must replace the terminal attachment in the
authoritative route scatter for a supported tendon; it must not be added on
top of the existing `J^T` force.

## First supported specimen

The first case is the right Achilles group:

- BodyParts3D `FJ1405` tendon and `FJ3360` calcaneus geometry;
- MyoSim `gaslat_r`, `gasmed_r`, and `soleus_r` source actuators (348, 349,
  and 369);
- Core/MyoSim `calcn_r` body 138; and
- the existing three-body tendon ownership record.

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

`NHTENDON1` will bind one or more authored route terminals to source bone
triangles. Each record needs the source route identity, distal body, triangle
identity, rest-frame barycentric coordinates, and the exact source/artifact
fingerprints used to register it. It must reject a mixed registration or a
terminal body other than the route's real distal attachment.

The native step remains one transaction:

1. Metal/Core evaluates the Hill/MyoSim route tension once.
2. The supported terminal is resolved from its named triangle at the posed
   body transform.
3. Core/Metal scatters the terminal traction into the articulated body using
   that point, replacing—not adding to—the old terminal scatter.
4. The same tension and attachment state drives the tendon rendering overlay.

The first release is an attachment-force law, not a deformable tendon
material. A later continuum phase can add a fibre or volumetric tendon state
only after its stiffness, damping, and failure/collision assumptions are
calibrated. BodyParts3D surfaces do not supply those parameters.

## Admission evidence

The native probe must retain, for each supported route, the source and
enthesis force/moment residuals, the distal body and triangle identities, and
the resulting whole-body generalized-force difference. A four-angle native
render must show the same resolved terminal on the named bone without using a
synthetic collar as evidence of mechanics.

Passing this milestone establishes a registered route-to-bone traction
representation for the selected tendons. It does not establish whole-tendon
continuum mechanics, tendon material calibration, contact, gait, injury
prediction, or clinical validity.
