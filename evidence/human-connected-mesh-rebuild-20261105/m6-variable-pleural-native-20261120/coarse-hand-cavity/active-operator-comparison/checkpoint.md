# Matched native Human solver schedules on Apple M6

These experiments use the exact 67-penetration release mesh and native
binary from e8ac916, the same initial pack, a 15.625 µs timestep, the
existing mechanical and vascular acceptance gates, and variable external
pleural pressure. All launch inputs and solver flags are pinned in the
receipts. The native body stage includes the coupled physiology
transaction and omits viewer presentation. Preparation and terminal
output time are included in process wall time.

| Native schedule | Roots | Body wall time | Recorded field result |
| --- | ---: | ---: | --- |
| Reference, budget 24 | 8 | 20.090 s | Reference |
| Assembled XYZ request | 8 | 20.321 s | Passes retained force and strain comparisons |
| Shared coupled compute encoder | 8 | 20.140 s | Passes retained force and strain comparisons |
| Completed graph work flag | 8 | 20.076 s | Passes retained force and strain comparisons |
| Newton budget 6 | 8 | 14.538 s | Passes retained force and strain comparisons |
| Reference repeat, budget 24 | 64 | 157.727 s | Matches the published 64-root control |
| Newton budget 6 | 64 | 113.999 s | Fails retained force and strain comparisons |

The graph flag affects the graph Schwarz builder when admitted. This
trial checks the flag's impact on this scene; it has no executed-kernel
counter for that particular path. None of the first three alternatives
showed a useful runtime gain here.

The exact 64-root repeat used the same diagnostic output settings as the
six-iteration experiment. It matches the published control on all
captured regional gates, including nodal mass and reference coordinates.
Its maximum accepted FEM position difference is 1.14e-16 m. The
six-iteration schedule cuts body wall time by 27.7% against this repeat
and still accepts 1 ms, but diverges starting at the root 14 Krylov work
counter. It misses 85 captured force comparisons above 1%, 210 strain
comparisons above 1%, and 181 strain comparisons above 5%. One right ACL
force channel changes by 0.03318 N. The maximum accepted FEM coordinate
difference is 4.692 µm. Both runs had identical connectivity, reference
coordinates and nodal masses.

The shorter schedule encodes 1,536 restart cycles versus 6,144, runs 601
Krylov columns versus 650, and encounters no linear-budget exhaustion.
The maximum active Newton starts per root are four and three,
respectively. The configured iteration budget changes the last-pass
schedule even when the observed number of active starts is below the cap.
The work counters alone do not prove equivalent accepted state. The
production launch keeps the 24-iteration budget and existing solver.
Its repeated body throughput is approximately 0.00000634× real time.

Both 64-root runs retain all 34 registered ligament rows, positive
terminal skin/fat cells and active external pleural pressure. The control
still has 67 initial point-in-bone pairs and less than 0.4 N of terminal
bed reaction. The terminal observation covers only 1 ms and does not
qualify supported equilibrium, zero crossings through motion, a complete
breath, or 0.1× with presentation.

The detailed raw logs remain on Numi. APFS compression verifies their
original SHA-256 before and after installation, preserves their bytes and
avoids publishing their routine endpoint dumps. Compact accepted traces,
selected physics evidence and reproduction commands are included here.
