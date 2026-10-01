# Public Open Knee comparison tools

Human/Matter owns native execution. These tools build an independent comparator
from pinned public MIT sources on Apple arm64; no FEBio account is required.
They do not create another runtime repository or make the hybrid source-equivalent.

Prerequisites: Xcode clang, CMake, an existing libomp installation, and Human's
Python environment with NumPy. The build does not install packages. Outputs must
be new directories. Sources, build recipes, commands, binaries and OpenMP are hashed.

```sh
.venv-mujoco312/bin/python tools/open_knee_reference/build_public_reference.py \
  --version 3.0.0 --output Build/new-public-febio3

PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.cli \
  open-knee-reference-case --open-knee Sources/open-knee-oks003 \
  --archive Build/open-knee-reference-20261001/doi-archive \
  --matter-root /absolute/path/to/existing-matter-worktree \
  --output Build/new-public-reference \
  --febio Build/new-public-febio3/febio-build/febio3 \
  --febio-config tools/open_knee_reference/febio3-superlu.xml \
  --comparison-version 3.0.0 --febio3-material-frames
```

The original archived solver was 2.9.1; public 3.0.0 is explicitly a different
version. The frame adapter records six changes: global parent fibre axes become
parent material frames with the elastic fibre expressed as local (1,0,0). The
prestrain uses the same parent frame. It retains all other mechanical values and
the immutable original deck. A translated deck can never report reproduction
with the original solver. Config files are frozen; without one the runner uses
`-noconfig`, avoiding an ambient unrecorded `febio.xml`.

The October 1 full comparison loaded 248,236 nodes / 844,287 solid elements and
assembled 495,960 equations / 19,903,986 stiffness entries. SuperLU factorization
crossed the explicit 18 GiB RSS cap on the 24 GiB Mac mini and was terminated;
**zero increments were accepted**. This is an unsuccessful baseline, preserved
alongside two earlier parse failures (2.9.0 lacks prestrain; unadapted 3.0 rejects
the legacy outer `fiber`). It is not evidence of a native memory or speed advantage.

The public build enables SuperLU 5.2.2 and Apple Accelerate, not MKL/Pardiso.
Unavailable MKL-dependent solvers/preconditioners explicitly reject execution.
GSL is not enabled: legacy fibre energy reporting is unavailable. In the source
GSL branch, the high-stretch energy also lacks a matching transition constant;
the prestrain wrapper does not override deviatoric energy reporting. These
builds cannot qualify energy-output equivalence. Matter's continuous energy is
checked independently against the force law.

## Constitutive equations

```sh
PYTHONPATH=src .venv-mujoco312/bin/python tools/open_knee_reference/compare_source_materials.py \
  --matter /absolute/path/to/build/numi-matter-fiber-check \
  --reference-build Build/new-public-febio3 \
  --output Build/new-material-comparison
```

The oracle links the actual unmodified public material implementations, including
the prestrain wrapper and isochoric generator. It checks its own analytic spatial
tangent against finite differences and exports first Piola stress and directional
tangent. The native executable exports its compiled symbolic expressions for the
same 486 cases: six tendons/ligaments, two menisci, cartilage; two fibre axes,
three sheared/volumetric deformations and nine directions. Metal execution is
also checked against the compiled native expressions. No whole-tissue equilibrium
or full-knee conclusion follows from this comparison.

## Cylindrical connector equations

Build with `--version 2.9.0` for the public connector comparator. It cannot run
the complete knee because that revision predates the required prestrain material.

```sh
PYTHONPATH=src .venv-mujoco312/bin/python tools/open_knee_reference/compare_source_joints.py \
  --deck Build/open-knee-reference-20261001/doi-archive/FeBio_custom.feb \
  --log Build/open-knee-reference-20261001/doi-archive/FeBio_custom.log \
  --matter /absolute/path/to/build/numi-matter-source-joint-check \
  --oracle /absolute/path/to/public-febio2/febio-build/febio-joint-oracle \
  --output Build/new-joint-comparison
```

Both operators receive the same normalized body poses for six source connectors
at 140 archived increments. Archived rounded poses select configurations; their
recomputed reactions must not be presented as high-precision reproduction of
the original logged reactions. The native operator retains finite source force
and moment penalties, prescribed/free coordinates and explicit multipliers.
It is not yet assembled into Matter's coupled knee solve.
