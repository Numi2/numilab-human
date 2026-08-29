# Anterior thorax continuum v1

## Result

`NHTHRC1` converts the non-conflicting exact source component 1 into a
deterministic tetrahedral continuum without changing its 723-vertex,
1,450-triangle anatomical presentation surface. Seven admitted `NHTENDON3`
terminal envelopes are mapped to that continuum through 28 four-node support
maps.

The compiler command is:

```sh
PYTHONPATH=src python -m numilab_human.cli \
  numi-human-anterior-thorax-continuum-payload \
  --registration Build/anterior-thorax-composite-v3.registration.json \
  --tendon-artifact Build/anterior-thorax-composite-v3-tendon \
  --output Build/anterior-thorax-continuum-v1
```

Two independent compiles are byte-identical. The generated payload is
`1,351,160` bytes with SHA-256
`599382747dd282a1a74b864cf8bff233c6ddf6524498cbdcd0be2d943844771b`.

## Volume and mapping gates

The source surface is closed and has an exact signed-volume magnitude of
`112.188360 mL`. Deterministic cell-centre voxelization converges as follows:

| Spacing | Cells | Connected components | Volume | Relative error |
| ---: | ---: | ---: | ---: | ---: |
| `3.0 mm` | `4,189` | `1` | `113.103000 mL` | `0.815272%` |
| `2.5 mm` | `7,169` | `1` | `112.015625 mL` | `0.153969%` |

Each occupied cube uses the same six-tetrahedron Freudenthal split. The final
mechanics mesh has `13,459` nodes and `43,014` positive tetrahedra. At the
declared `1,000 kg/m3` provisional density its mass is `0.112016 kg`.

The exact source presentation surface is retained separately and follows the
continuum through four-nearest boundary-node displacement transfer. The
support distance is `0.738 mm` RMS and `4.306 mm` maximum. This is a
displacement map: the exact source surface remains exact in the rest pose; it
is not replaced by the voxel boundary.

The generated posterior support band maps to `190` continuum anchor nodes.
That band is an explicit numerical assumption, not a source-authored
bone/fascia weld.

## Component 17 exclusion

Component 17 is deliberately not tetrahedralized. The v2 source receipt gives
that same exact component two incompatible roles:

- `EO4_l` uses it as an unresolved anterior-thorax composite surface.
- `EO5_l` and `IO5_l` use it as the exact source fallback for the registered
  left tenth rib (`FJ3225`).

Its thin closed volume also fragments under ordinary cell-centre sampling from
`5.0 mm` through `1.0 mm`. Promoting it as deformable tissue would therefore
overlap an existing rigid-rib owner and conceal an unresolved bilateral
cartilage/rib correspondence. `EO4_l` remains on its existing exact
`NHTENDON3` source-surface law until that correspondence is established.

## Force-ownership gate

The payload encodes:

- production tissue-owner fraction: `0.0`;
- bounded deformation-probe load fraction: `0.10`;
- direct joint torque: forbidden;
- simultaneous rigid and tissue force ownership: forbidden.

MyoSim's route `J^T` remains the sole production rigid generalized-force
owner. The continuum may become an owning two-way boundary only when the same
transaction removes the matching endpoint share from that rigid route and
returns accepted anchor reaction into the rigid solve. A one-way Matter load
or a rendered displacement is not that closure.

## Material boundary

The source `ribcage_s` component does not identify one tissue. It must not use
the existing pectoralis-major fascia GOH law merely because that law is
available. `NHTHRC1` therefore supplies geometry, mass discretization,
surface/attachment maps, anchors, provenance, and ownership state, but no
biological constitutive identity. Material classification and calibration are
separate gates.

This is a connected and converged continuum input, not yet a calibrated
anterior-thorax material solve, breathing model, contact qualification, or
two-way muscle-tissue-rigid simulation.
