# Pressure on the registered pelvic surfaces

On SSH Numi, the integrated scene completed nine successive 2 ms steps with variable pleural pressure. All nine endpoints passed exact transverse triangle checks in the six gluteal muscle–bone banks. An independent terminal check found zero boundary pairs, zero intersecting volume, and zero interior samples among 26,981 forward queries. The two muscle domains had a minimum terminal deformation determinant of 0.9930101. The rejected continuation preserved the completed replay's material state and surfaces byte for byte.

The local pressure solve now uses registered material vertices and cortical triangles for finite-area rows and forward point queries. Candidate generation accounts for directional rounding. Acceptance checks the physical gap, triangle crossings and free force residual. A crossing on an otherwise converged candidate requests another mechanical correction. A clipped feature that disappears releases its previous compressive load. The selected solver damps pressure and displacement updates together only on a side that has lost support. The original tolerance and force transpose remain in force.

## Measured limits

The 48-step request accepted 18 ms. The first rejected candidate at 20 ms had zero transverse pairs and zero measured penetration. Its left free force residual was 296.018 µN against a 10.768 µN limit. Its positive contact gaps were at most 14.1 nm on loaded rows using the release metric. The static bed supported 744.12 N at the ninth endpoint for a 97.132 kg model whose center of mass was descending at 0.115 m/s. Support was still settling. Whole-body crossing clearance, an entire breath or heartbeat, and the 16-second motion requirement remain open.

On the dedicated nine-step replay the native timer measured 5.1164 s wall (0.003518×), including a 1.2067 s GPU command span. Measured from application launch the replay took 13.7794 s (0.001306×). These short-run measurements include startup effects. Presentation qualification is pending. Hardware was Apple M6 with 12 GPU cores, Metal 4, macOS 27.0 build 26A425.

## Retained comparisons

The corrected pressure builds were checked against an independently computed overlay of the actual emitted geometry, including touching, sliding, reversed orientation and a narrow resolvable sliver. The compatibility run reproduced the previous legacy state and surfaces byte for byte. The failed forward-site mapping and its corrected authoring receipt are both retained. Two runs stopped at the disk watchdog before providing a native result. The diagnoses of those runs come from their successful retries. The other native trials distinguish certificate failures from actual geometric crossings. The replay script takes 9 and 48 as the short and long horizons for this checkpoint.
