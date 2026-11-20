# Matched skin source and execution-shell clearance on Apple M6

The native diagnostic replay emits the detailed weighted skin and the compiled
physical shell at the same initial pose. The shell, physical bones and native
input pack match their pinned predecessor byte for byte. The replay accepted one
15.625 µs root with the existing coupled solver and variable pleural-pressure
configuration. Its process took 19.33 seconds including startup and export.

Exact Float32 tests against 469,136 triangles across 86 physical bone banks
find 1,700 detailed skin crossings involving 555 source faces. The largest
source conflicts are in the hands. The coarse outer layer has 1,380 crossings
across 28 bone banks that the detailed outer skin clears. The counts use each
surface's own face IDs. A total of 17,878 coarse pairs spans the outer skin,
the skin/fat interface and the inner fat boundary. Only intersections at the
captured pose are measured.

The conditioned hand fit reduced the source count to 1,009. It created 793
new face-pair identities, with zero new skin self-intersections. The new
identities include moving intersections within the already affected bone
banks. This leaves the current detailed skin unchanged in native execution.

Four skull experiments pair each refined tetrahedron and surface triangle
with its original parent. Bisection added 162 nodes and 604 tetrahedra.
Undisplaced bisection preserved all 17,878 original shell-face/bone-face
identities, even though the triangle-level pair count rose to 18,123.
The mass map splits each original tetrahedron's mass by its children's
deformed Float32 volumes and retains its material. The test reproduces
13.93967896346048 kg with positive cell determinants and no new local
surface self-intersections. Native density, donor registration and the
trajectory gates still apply.

| Skull candidate | Targeted outer pairs | Whole-body pairs by original face ID |
| --- | ---: | ---: |
| Original | 702 | 17,878 |
| Edge-only fit | 702 | 17,887 |
| Connected fit to the detailed skin | 403 | 17,512 |
| Connected fit toward the bone envelope | 468 | 17,289 |
| Connected fit with independent layer depths | 400 | 17,301 |

Of the 91 columns sampled against both registered surfaces, 86 have less
room than their current projected tissue thickness plus 50 µm. Median
available space is 5.59 mm and median current projected thickness is 12.25 mm.
The independent-layer trial reduced local error and retained fixed support
nodes, but still creates new bone pair identities in the same skull bank.
The retained geometry and solver inputs remain at their pinned revisions.
A locally rebuilt thickness field and conservative material remap are needed.

The repaired full-resolution reference separately has zero of the 125
inverted and 154 low-quality cells and its independent allocation conserves
13.94458033634514 kg. The native scene still uses the distinct coarser shell.
The 34 registered ligament rows passed the short native terminal audit.
Supported shoulder loading, body-bed equilibrium and continuous crossing
qualification still need a full-scene test. The last matched 128-root
control advanced 2 ms at 0.00002028× body time and carried 0.400701 N
against 952.864 N of body weight from the fixed bed.

## Reproduce

Use the retained experiment tree on SSH Numi with the exact input digests in
the receipts. The preparation programs use the existing Numi tooling for
the native launch, mesh types, source identities and exact predicates. The
registered source dump and repair proposals have byte-verified private
recovery assets. No raw source geometry is included here.

~~~sh
python3 audit_source.py "$matched_native_output" "$audit_output"
python3 fit_source_faces.py "$matched_native_output" "$hand_screen_output"
python3 refine_clear_banks.py "$matched_native_output" "$head_screen_output" --connected --fit-layer-thickness
~~~

The 100 analytic determinant derivative comparisons, synthetic orientation
check, signed ray check and layer-constraint test ran on SSH Numi before
repeating all four skull screens with the retained Python source. No human
physics setting is changed by these preparation tools.
