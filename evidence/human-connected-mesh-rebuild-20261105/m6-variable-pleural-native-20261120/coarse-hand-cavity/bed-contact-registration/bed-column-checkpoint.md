# Physical bed contact registration

Device Apple M6 on SSH Numi. Parent checkpoint 54b9809750c7c6a2234739477c2ebccf58ffaab3.
All native trials retain the coupled tissue, articulation, ligament,
circulation and variable external pleural-pressure paths. The fixed bed plane
acts through physical contact. The initialization posture, time step and
solver acceptance tolerances are unchanged. These are startup experiments.

The first fit placed two nearby outer coarse nodes at exact registered skin
points 17764 and 21161. This reconciled sampled vertical deficits of 2.632 and
1.947 mm. It required transverse moves as well, up to 13.63 mm in total. The
complete connected patch conserved the two layer volumes, with minimum
emitted relative tetrahedron shape quality of 0.88996. The emitted whole
skin/fat mass changed by -5.84e-8 kg. The existing parent partitions all
passed. The exact static Float32 audit found zero new intersections across
the 382 changed faces and the remainder of the shell. The initial
skin-to-bone audit still found 71 point penetrations.

Three physical outer nodes sat approximately 24 nm above the bed, with the
gravity projection of the whole-body center of mass inside their triangle.
This establishes a geometric support possibility. It does not measure load
balance. The native trial rejected its first root with vascular line-search
status 10 at row 45 and a mechanical residual of 9.658 versus a limit of
0.00012072. Moving the fixed plane to the existing 10 µm barrier-release
distance lowered the mechanical residual to 0.06646, still beyond its
0.0000125 limit. Row 45 rejected again. No physical time advanced in either
run.

The unchanged parent at the same 10 µm barrier-release distance accepted
one 15.625 µs root. The certified mechanical and vascular norms were
8.339e-6 and 2.182e-8. The bed reaction was 20.63 µN compared with a body
weight of 952.86 N. APFS ran out of space after this accepted step during
final output. No successful exit or throughput was recorded for that
control. The compact excerpts retain the source log hashes. Raw packs
were retired after independent SHA-256 verification of streamed restores
from the private release recorded in each recovery receipt.

Pinning the deep fat anchors while attempting exact XYZ point placement
failed the static feasibility check in both regions. A separate, smaller
repair fits only the depth of nearby physical skin nodes, preserving
in-plane registration at those nodes and holding every deep fat anchor
fixed. Its preflight retained positive tetrahedra and exact parent
partitions. See the additional native trial and its evidence below.

The depth-only fit moved 22 outer/interface nodes by at most 0.984 mm.
All deep fat anchors stayed fixed. Both materials conserved fitted volume,
and the emitted mass changed by +4.07e-9 kg. Minimum relative emitted cell
quality was 0.91878. The native Float32 shell audit found no intersections
touching the 98 changed faces and no new point/bone penetrations. The 71
pre-existing point/bone penetrations remain.

At the existing barrier-release clearance, the depth-only fit completed
four accepted roots in 29.459 s process wall time. The body stage used
10.136 s for 62.5 µs of simulation, or 0.00000617× real time. All 34
registered ligament rows participated and the terminal ligament audit passed.
The last root supported 0.109 N through three physical contact rows. Across
a second run with identical starting state and a verified clone of the
initial pack, all 64 roots passed both physical residual gates. The last root
at 1 ms supported 0.387 N through those three rows. All 28,632 shell
tetrahedra had positive terminal Jacobians (minimum 0.94900), with zero
root assistance. The 64-root run encountered ENOSPC while writing the
remaining terminal diagnostics and did not produce an exit or elapsed-time
receipt. The accepted trace and complete shell cell audit survived.
External pleural loading was active, and the short accepted trace spans
-490.332520 to -490.332489 Pa. This transient does not establish behavior
over a breath.

The four-root process validates the new initialization geometry and the
64-root trace demonstrates a longer startup prefix. The torso and limbs
are still releasing toward the bed. Balanced support, full-scene crossing
clearance, 16 seconds of continuous motion and 0.1× remain open.
