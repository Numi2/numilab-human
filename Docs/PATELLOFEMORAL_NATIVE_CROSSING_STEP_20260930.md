# Native patellofemoral crossing admission - 30 September 2026

The [source-bound loop](PATELLOFEMORAL_INTERSECTION_LOOP_20260930.md)
contains 18 patella–femur triangle crossings in each compiled knee. To test
the owning Matter contact path at small scale, the
[fixture](media/patellofemoral-crossing-fixture-20260930/receipt.json) extracts
the **two exact authored tetrahedra** adjacent to the first left crossing pair
from the pinned NHKNEE1 payload. It preserves their Float32 SI positions and
tetrahedron order. This is one diagnostic pair, not the full knee.

The native Apple M4 Matter probe, built from Numi Lab `coupled` commit
`524274a`, runs two FEM
objects with generic deformable contact enabled, zero gravity, a 1 µs step and
an explicitly **synthetic silicone material**. Its accepted step has status 0,
one completed microstep and five active deformable contact histories. An
independent exact face check still finds a crossing: its segment changes from
**68.260 µm** initially to **68.050 µm** afterward. The native result repeats
byte for byte. A control translated 3 mm in +Y has no face crossing and no
active deformable-contact history. The generic `contact_count` field is zero
for both runs; the accepted deformable histories are the relevant contact
readback here.

The [exact native-step audit](media/patellofemoral-crossing-fixture-20260930/native-step-audit.json)
therefore has status **`failed_initial_crossing_clearance`**. Native step
acceptance and contact activation have not established noninterpenetration.
The full source knee remains blocked for loaded-contact qualification. This
fixture supplies neither physiological cartilage properties, support, load,
pressure, energy closure nor a sustained accepted knee state. It makes no
clinical-anatomy claim.

The contact owner needs an initial-intersection policy that either provides
source-bound separating constraints and proves a crossing-free accepted state,
or rejects the intersecting input before it can be counted as a valid loaded
step. An unsigned proximity barrier alone does not encode the crossed surface
topology. The 18-pair bilateral loop must then be checked on the complete
cartilage meshes, including replay, rollback, reaction and energy.

Rebuild the fixture and audit the captured native runs:

```sh
cd /Users/home/numilab-human
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m tools.build_patellofemoral_crossing_fixture
PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m tools.audit_patellofemoral_crossing_native
```

The native probe source is
`numi-lab/matter/tools/patellofemoral_crossing_probe.mm`; build target
`numi-matter-patellofemoral-crossing-probe`. Run it with the fixture text,
or append `0 0.003 0` for the separated control. The captured executable
SHA256 was `4265137b71c2ab52a6e421830ab7a5a44856757d1bf78777454289e296b5168e`;
the metallib SHA256 was
`1fa5eeb90d506dd89ba10905a9ed543a2376342a13b2fd3ed9bbda98e96931ac`.
