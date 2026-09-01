# Anterior abdominal-wall continuum v1

## Result

`NHTHRC1` now supplies a live bounded force owner for seven exact
anterior-thorax attachment boundaries: bilateral rectus abdominis insertions,
bilateral EO2 origins, right EO4, and bilateral IO4 origins. The exact pinned
source component remains the 723-vertex, 1,450-triangle presentation surface.
Its internal Matter envelope has 2,556 nodes, 5,424 positive tetrahedra, 83
traction nodes, and 126 fixed nodes.

The compiler command is:

```sh
PYTHONPATH=src python -m numilab_human.cli \
  numi-human-anterior-thorax-continuum-payload \
  --registration Build/anterior-thorax-composite-v3.registration.json \
  --tendon-artifact Build/hand-enthesis-fixed-cluster-v2-tendon \
  --output Build/anterior-thorax-continuum-v1
```

Two independent compiles are byte-identical. The current payload is 250,208
bytes with SHA-256
`066d6ada2d0680df43351fa08f3fc1e65a80387f757a72173f01bc38edf9f065`.
It pins the current NHTENDON3 payload SHA-256
`a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555`.

## Efficient mechanics envelope

The exact closed source volume is `112.188360 mL`. A 5 mm cell-centre sample
contains 887 cells in 15 sub-voxel components because several exact source
necks are narrower than one cell. The compiler joins them with 17 deterministic
minimum six-neighbour corridors. The admitted envelope therefore has 904 cells,
one connected component, `113.000000 mL` volume, and `0.723462%` relative
volume error.

A coarser 6 mm check requires 37 repair cells and has `9.166405%` volume error.
The 5 mm envelope is both more connected-efficient and more accurate. Every
repair cell is counted in the volume error; it is not hidden as source volume.
This replaces the earlier 2.5 mm 43,014-tet envelope, whose one-step Metal
smoke exceeded the bounded qualification interval, while preserving the exact
surface and attachment maps.

The exact surface follows the FEM through 723 four-node maps. Seven terminal
patches use 28 additional four-node maps. Component 17 and `EO4_l` remain
excluded because the same exact component also owns a left-tenth-rib fallback;
overlapping rigid and deformable ownership is forbidden.

## Force ownership

The payload declares a `0.10` production owner fraction. In endpoint mode the
opposite muscle terminal is the load witness and the exact torso-side endpoint
is the replaced anchor endpoint. The live Apple transaction:

1. reads the borrowed NHTENDON3 terminal-force buffer;
2. distributes 10% of the opposite-terminal force over the four sample maps;
3. removes exactly 10% of the matching torso endpoint's source `J^T` share;
4. advances the fixed band with the source torso body;
5. projects accepted fixed-node reaction through that body's Jacobian; and
6. commits or restores Human and Matter state together.

The other terminal remains source-owned. Direct joint torque and simultaneous
rigid/tissue ownership are forbidden.

## Material boundary

The source component is an unresolved abdominal-wall composite, not a named
single tissue. The starting law uses the `1.12 MPa` median elastic modulus for
human full-thickness abdominal-wall composite specimens reported in the
[human abdominal-wall characterization](https://pmc.ncbi.nlm.nih.gov/articles/PMC10604332/).
The 10% owner uses 10% stiffness (`E = 0.112 MPa`) to preserve the first-order
force/stiffness strain scale. With an explicit `nu = 0.45` assumption this gives
`mu = 0.0386207 MPa` and `K = 0.373333 MPa`.

This is an isotropic compressible neo-Hookean fallback. Human EO, IO, TrA, and
rectus tissues are layered with different fibre directions
([architecture evidence](https://pmc.ncbi.nlm.nih.gov/articles/PMC3017737/)),
and linea alba/connective tissue is nonlinear and anisotropic
([biaxial human evidence](https://pubmed.ncbi.nlm.nih.gov/27367944/)). The
current material is therefore not subject-specific or a calibrated directional
failure law.

## Apple M4 Pro qualification

At a `10 us` step with a 2% selected increment on source muscles 22, 23, 187,
189, 195, 199, and 207, the four-step live run measured:

- `78.4833 N` assigned tendon-force L1;
- `3.32478 N` fixed-node reaction L1 and `0.062026 N` maximum node reaction;
- `0.249343 mm` maximum free-node displacement;
- `Jmin = 0.956072` and two FGMRES iterations;
- maximum Human difference from source-only `J^T` of `1.49e-8` in q and
  `3.02470e-4` in v;
- all 832 tendon transfers, including 641 distributed envelopes and 191 exact
  point laws;
- bitwise replay and verified downstream-rejection rollback; and
- `12,366.76 ms` for the complete four-step coupled transaction on Apple M4 Pro.

The one-step smoke also passed with `78.4095 N` assigned force, `1.28891 N`
reaction L1, `0.025625 mm` displacement, and `Jmin = 0.995378`.

## Evidence boundary

This closes a bounded two-way mechanics path for seven existing exact
attachment surfaces. This v1 slice does not classify component 1 as one
biological tissue, model breathing, resolve abdominal-layer sliding/contact,
calibrate a subject, or provide clinical validation. The generated fixation
band, connectivity corridors, isotropic material reduction, and 10% share
remain explicit assumptions. The later `NHFASC4` owner closes the fourteen
body-7 abdominal terminal transactions at the same bounded evidence level; it
does not retroactively turn those terminals into bone or make this anterior-
thorax composite a validated tissue segmentation.
