# Posterior fat-to-muscle registration and conservative mass

The released native scene has a measured discontinuity in its intended
bed load path. On SSH Numi, 130 posterior shell face samples within
5 mm of the bed pointed inward to the gluteus maximus. The existing fat
columns are about 9.9 mm long there. The nearest represented gluteal
surfaces along those bed-normal paths are another 12.4–23.2 mm inward,
with a median of 18.0 mm. The source-to-shell distance is specific to
this mixed-source reference and pose. A CT study measured regional
subcutaneous thickness across 298 pelvic scans. An independent
weight-bearing MRI study measured deformation in both gluteal fat and
muscle. [CT thickness study](https://pubmed.ncbi.nlm.nih.gov/17521050/)
[Weight-bearing MRI study](https://pubmed.ncbi.nlm.nih.gov/28987874/)

The geometric audit uses the exact startup shell and mechanical bone
inventory, including 469,136 bone triangles. Muscle uses the registered
NHTISS4 source and the first accepted body transforms at 15.625 µs. The
86 shared transform positions differ from recorded startup by at most
0.37 µm. The audit retains typed source triangle IDs, barycentrics and
pose comparison. The muscle surface is a reconstructed source. Its
presence supplies geometric evidence and does not provide a transverse
force path in the existing native muscle owner.

The first direct extension inverted 27 cells. Constraining the affine
tetrahedron determinants in both coordinate frames cleared these
inversions, after which an exact audit found 152 new muscle intersections.
The second fit added constraints from the projected overlaps of those
pairs. The fit with typed mass bounds moves 187 deep nodes while retaining
exact outer and dermal interface coordinates and cell ancestry. All
28,632 tetrahedra remain positive. The worst emitted determinant ratio
is at least 0.44999, and the minimum global normalized shape matches
the original. Fifty-one of 68 near-bed target vertices approach their
registered muscle points within 0.5 mm. The exact audit finds zero new
muscle, shell, other continuum, or bone pairs touching the changed
faces. Seven existing pairs against bone bank 128/19 remain. This
static check covers affected faces, with no intervening-motion or
whole-scene clearance inference.

Holding density fixed on the first clearance fit would add 0.160 kg.
Its original typed mass budgets are at least 28.2 g short at minimum
allowed density. Constraining the fit against those budgets removed
the deficit. Sparse bounded flow through the existing 75 mm locality
radius then mapped the original source masses using unchanged material,
anatomical region and articulated donor identities. The integer flow
only selects a sparse support. The real-valued solve and independent
checks validate exact source mass closure and allowed target density.
The output has 13.939679839 kg in the existing reference mass ledger.
See the transport receipt for source residual and moment diagnostics.

This candidate remains an offline reference. It exceeds the old
whole-body 10 mm adipose thickness limit locally and has no native
contact attachment to the gluteal surfaces. Native admission needs a
regionally validated thickness field and a reaction owner that transfers
forces through the underlying anatomy. Existing native physics and
the published launch remain unchanged. Shoulder ligaments under
settled load, equilibrium, zero crossings through 16 seconds with
variable pleural pressure, and the throughput targets are unverified.

## Reproduction on SSH Numi

Place these files beside the earlier coarse-shell trial folders and
use the retained Python preparation environment.

```sh
T=$(cd .. && pwd)
R="$T/loaded-shell-seed-039/native-inner-fit-022/candidate-128"
X="$T/thickness-rebuild-034/inputs/physical-bone-initial.bin"
M="$T/../resting-clearance-and-atrium-006"
export PYTHONPATH="$T/../resting-surface-clearance-001/src"
export OPENBLAS_NUM_THREADS=1
python3 -m unittest -v test_authoring
python3 audit.py "$R" "$X" reproduced-witnesses.json --muscle-reader "$M"
python3 extension.py "$R" reproduced-witnesses.json "$M" reproduced-extension.json \
    --candidate reproduced-reference.npz \
    --exclude-crossings-from posterior-candidate-all-crossings.json \
    --preserve-typed-mass-with "$T"
python3 map_mass.py "$R" "$T" reproduced-reference.npz reproduced-transport.json \
    --mass reproduced-mass.npz --solve-transport
python3 check_candidate.py "$R" "$X" reproduced-reference.npz "$M" "$T" \
    reproduced-clearance.json
```

## Anatomical asset attribution

BodyParts3D, Copyright © 2008 The Database Center for Life Science,
licensed under CC Attribution-Share Alike 2.1 Japan. The retained
BodyParts3D 4.0 triangle data were registered, reduced and adapted for
this mixed-source shell. The adapted geometry and derived anatomical
witness coordinates in this directory are available under the original
attribution and share-alike terms. The body pose and mass mapping are
reference constructions. These files contain neither the full MyoSim
input asset nor the original anatomical mesh archives.

[Historical data license and attribution](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20100816/README_e.html)
[CC BY-SA 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/)
