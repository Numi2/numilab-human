# Rodero-04 source activation transfer

The `numi human rodero-case04-source-lat` command imports the Healthy Rodero-04
activation-time field published with the [His-Purkinje/CardioMechanics workflow
record](https://zenodo.org/records/21934382). It pins all three source archive
hashes and the MATLAB conversion script, checks that the script converts LAT
from milliseconds to seconds, and verifies the source case settings bind the
Healthy LAT field to the paired TetGen mesh.

The source VTK contains 103,509 points, 543,571 cells (487,583 tetrahedra and
55,988 triangles), with one LAT value for each VTK cell. The TetGen mechanics
mesh contains 487,583 tetrahedra. The importer confirms that the point IDs use
the same coordinate order within `1e-5 mm`, then maps every TetGen element to a
unique source VTK tetrahedron by sorted node connectivity. The current pinned
archives produce 487,583/487,583 exact connectivity matches. Activation values
are written in TetGen element order, alongside a one-based source-cell map and
an immutable JSON receipt.

Run it from the repository root with the three pinned archives available:

```sh
NUMI_HUMAN_PYTHON=.venv-mujoco312/bin/python numi human rodero-case04-source-lat \
  --geometry-zip /path/to/data.zip \
  --cardiomechanics-zip /path/to/CardioMechanics.zip \
  --matlab-tools-zip /path/to/MatlabTools.zip \
  --output-dir Build/cardiac-rodero04-source-lat
```

This is a source-data transfer for Rodero-04. It does not repair the current
case18 source activation mismatch, provide voltage or ionic dynamics, run a
native electrical step, couple electrical activation to mechanics, or qualify
a heartbeat or clinical electrophysiology. The Zenodo record snapshot reviewed
for this import did not state a data license; the raw archives are therefore
not redistributed by this change.
