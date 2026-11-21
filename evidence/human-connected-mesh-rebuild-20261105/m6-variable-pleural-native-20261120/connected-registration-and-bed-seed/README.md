# Connected reference registration and a physical bed-pose trial

The starting reference contains 311,020 skin/fat nodes and 1,448,804 cells.
The pinned mesh emitted in Float32 has zero inverted or zero-volume cells,
zero cells at or below the signed-volume floor, and zero cells below
the normalized determinant floor. Native control 026 replayed
13.939678963460478 kg of mapped skin/fat mass, admitted all 34 ligaments and
accepted 16 steps through 0.250 ms. It retained 14,179 nonocular skin-to-tissue
triangle pairs at its endpoint, with zero terminal bed load. The control
used variable pleural-pressure coupling. Its accepted samples show
−490.332489 Pa at output precision over the short trace.

## Source anatomy

Six muscle banks with prior initial-pose coverage contain 4,749 exact
binary32 muscle/outer-skin triangle pairs on the starting geometry. A
witness-local connected graph solve did not improve any bank within the
shape bounds. A smooth, low-dimensional spline fit reduced the total to
4,371 across the independently tested banks. The candidate limit is 3 mm
maximum displacement, 1% aggregate signed-volume change, a per-face area
ratio from 0.8 to 1.25, and no degenerate or flipped faces. Both longitudinal
end bands are fixed. Direct enthesis validation remains future work.

The exact nearby checks include the inner and deep shell boundaries,
bones, neighboring muscle banks, and adjacent organ surfaces. All five
locally improved proposals create new neighboring triangle pairs.
No candidate passed the complete crossing gate. No source assets changed.

The right brachioradialis and flexor pollicis longus banks, 109/135, share
coordinates in 7,074 of 7,276 intersecting triangle pairs. The left pair,
110/136, shares no exact coordinate in 259 witnesses, all with vertex
excursions above 1 µm on both sides of each opposing face plane. These
counts distinguish coincident-coordinate pairs from clearly transverse
pairs. Both classes remain in the reports until there is a validated
common interface. A shape-preserving connected repair must account for
these interfaces and the inner and deep boundaries together.

## Geometric bed seed and native result

The rigid fit selects the lowest tilt from the lower convex hull for
which the contact triangle contains the projected articulated COM proxy.
Its 0.143974° rotation and 1.99934 mm downward motion are authored only at
initialization. Three nodes were within 10 µm of the bed, spanning
0.449344 m along the body, versus one node in the unchanged release.
The test does not establish a loaded equilibrium.

The native trial used the registered physical outer surface and the
unchanged support and acceptance rules. It passed all 118 enabled joint
ranges, admitted 26 fascia sources and all 34 ligament rows, then rejected
the pinned outer, inner and deep bindings at donor row 1. The emitted
current-frame origin and matrix changed and their rest values matched.
The run installed no volume and accepted zero physical steps. The
compact launch, failure and empty accepted trace are included.

The independent full-mesh screen applies the same rigid transformation
in Float64 and repeats the original quality evaluation after rounding
each world coordinate to Float32. The Float64 transform clears both
floors. The simulated Float32 emission has 295 inverted or zero-volume
cells, 334 at or below the volume floor, and 1,331 below the normalized
determinant floor. The counts overlap. Maximum coordinate rounding is
61.5 nm. There are 38,830 cells with more than 1% volume error. The
skin/fat aggregate signed volume changes by less than 3e-10 m³, which
does not reveal these local failures. The trial indicates that changing
startup pose requires resolving world-coordinate sensitivity as well as
rebinding the donor data.

## Reproduction and storage

Run the Python screens from this directory using the retained Numi
preparation environment. Pass the control launch directory
`floor-native-live-parent-026` to the graph, spline, forearm and bed fit
scripts. The scripts use the retained exact predicate package and pinned
mesh files. Run `audit_bed_pose_volume.py` with the
`constitutive-witness-repaired-tetgen` directory and `bed-pose-fit.json`.
Each recorded numerical report matched a second run byte for byte.
The native wrapper is optional and uses the exact pose receipt and
matched library. It uses a temporary capture volume and keeps the
external system-volume watchdog.

Three closed outputs were compressed and checked against the original
lengths and SHA-256 digests. Two already published private-release archives
were streamed from GitHub and verified before their local downloads were
retired. These operations recovered about 809 MiB. Two inaccessible replay
paths were also restored from verified compressed clones. Their extents
were already shared and restoration is not counted as recovered space.

Supported equilibrium, sustained variable-pressure motion, continuous
whole-body crossing clearance and 0.1× throughput remain open.
