# Native muscle registration replay

Tested on SSH `numi` with the Apple Metal runtime and hash-pinned full-volume
inputs in `launch.json`. This retained candidate uses the mixed-source
reference adult. The two changed BodyParts3D muscle surfaces are right external
oblique (67) and temporalis (164). This change preserves the original source
topology, influence records, pinned positions and eight parent assets. It
changes passive registered surface positions and normals. Its replay verifies
that these edits did not change the active muscle, tissue, ligament, or
physiology force path.

## Native result

All 16 physical steps accepted. They cover 0.250 milliseconds, with a 15.625
microsecond step. The process took 337.11 seconds including preparation and
export (7.416×10⁻⁷ simulated seconds per wall second). The body stage took
195.64 seconds. The runtime pressure coupling was enabled. The 16 samples all
report −490.332489014 Pa, so this prefix does not qualify variable pleural
pressure. The signed terminal bed reaction is 0 N. This is an unsupported
release from a bed-adjacent pose.

Four physical traces, full terminal body poses, the installed skin/fat
mapping, terminal continuum certificates and all 34 ligament rows match the
control exactly. The two revised surfaces are the only changed positions
among 879 compared geometry banks. Retained skin and back positions are
unchanged at this endpoint.

| Native terminal measurement | Control | Candidate |
| --- | ---: | ---: |
| Whole-body nonocular skin-to-tissue triangle pairs in the retained audit | 14,179 | 14,100 |
| Right external oblique skin triangle pairs | 722 | 659 |
| Temporalis skin triangle pairs | 466 | 450 |
| Right external oblique sum of transverse pair lengths (mm) | 458.255 | 392.620 |
| Temporalis sum of transverse pair lengths (mm) | 193.834 | 184.866 |

Both edited surfaces have zero self-intersections. The exact changed-face
audit found zero new neighboring tissue intersections and zero new skin
witnesses outside the previous affected vertex rings for these two banks.
Their new pair counts within those rings were 401 and 66. The largest
individual oblique intersection segment grew from 5.155 to 5.304 mm. The
numbers are triangle-pair and segment measurements. Qualification requires
eliminating the crossings and measuring physical load, strain and contact
through motion. This candidate remains opt-in evidence.

## Initial support footprint

The same native registration places only one of 76,153 outer skin nodes
within 10 micrometers of the bed. At this cutoff the articulated center of
mass projects 100.25 mm from the possible contact point. For a normal-force
support hull to first contain the articulated center of mass projection, the
cutoff needs to reach 0.9821 mm above the initial plane. This calculation
uses the 82.716 kg articulated mass alone. The remaining finite-element mass
still has to be included in any full equilibrium calculation. The native
initializer requires a fresh FEM bed release and does not admit its legacy
rigid support seed. Additional body contact needs to develop by solving
the native loaded tissue equilibrium with a single bed-force owner.

## Reproduction

The source-coordinate edit, Float32 preflight and launch wrappers use the
retained SSH Numi working paths. `source-replay.json` pins the original
surface, derived binary and fitted source coordinate deltas. The first
launch with the relocated anatomy receipt stopped before physical execution
because its relative cardiac map was absent. The recorded run hard-links and
verifies the three original sidecars. `retained-inputs.sha256` pins the
local geometry and measurement routines. The native captures stayed on Numi
and were transparently compressed after their byte-identical readback was
verified. Cumulative allocated storage savings were 753,012,736 bytes,
including older preparation flows. No generated bulk geometry is committed.

## Bed force and moment preflight

The paired skin and fat role packets match every registered TetGen cell and
the exact Float32 node coordinates. Their cell masses link back to the
verified native paired-flow receipts. Skin contributes 3.452570445 kg and
fat contributes 10.487108518 kg. The source pose accounts for 99.51% of mass
in its articulated and skin/fat first moment. The full compiled body mass is
97.13194745 kg, with weight 952.8644 N at 9.81 m/s².

The one skin node within 10 µm of the plane is 87.465 mm horizontally from
the combined mass projection. For that node alone to hold the body in static
balance, the other 0.476 kg of continuum tissue would have to be centered
17.76 m horizontally from the support point. This rules out point-supported
static balance at this footprint and identifies why using the existing
release pose as a restart cannot establish resting equilibrium. The computation
uses a geometric contact cutoff. It does not assign bed reaction, recompute
tissue deformation, or certify that the supported state has been reached.

Run `check_bed_mass_moment.py` after `check_bed_footprint.py` on the pinned
Numi reference. The report retains both geometry and mass identities.
