# Witness-based skin and fat clearance on Apple M6

The native initialization now clears six deep-fat point-in-bone pairs,
two in registered bank 128/19 and four in 150/5. All 61 remaining pairs
are on the deep fat boundary. The same initial audit covers 86 eligible
closed bone banks and both outer and deep boundaries, with no new
penetrations or ambiguous pairs. The 400 changed shell faces have no
new intersections. The physical bed's three registered outer points and
their deep fat anchors remain fixed during fitting.

A 3 mm component displacement box could not reach ten measured
witness tangent planes, even before applying tissue constraints. The
new offline fit derives a component bound from the measured depth and
normal. It keeps the existing nonlinear shape, Jacobian, thickness,
volume and exact refinement-ancestry checks. This is a reference mesh
edit. Native contact, constitutive laws, timestep, pressure loading
and acceptance remain unchanged.

The fit moves 98 emitted nodes. All parent and child tetrahedra stay
positive, and no explicit refinement child exits its parent. The
minimum changed-cell relative Float32 shape quality is 0.88999993.
The paired native mass mapping changes the skin/fat nodal sum by
+1.2200e-7 kg and retains the 97.13195069110952 kg reference body
budget. The dominant connected 61-witness solve reaches 75% of its
modeled target. The full-clearance proposal fails the shape constraint
at skin tetrahedron 13137 and is discarded. Two disjoint patches pass
all checks and produce the six native clearances above.

The first run completed initial physical preparation and registration,
then failed at visual-pack streaming before a physical step. An older
unrotated startup pack and its exact alias were retired only after an
independent GitHub download restored the original SHA-256. The
recovery receipts preserve the pack identity, asset and original
paths. The retry uses the exact same binary and real contact models.
The unchanged production solver retains the 24-iteration budget.

The four-root and 64-root native runs both completed. The longer run
accepted 1 ms with matching physiology time and zero root assistance.
All 28,632 shell cells stayed positive at the terminal capture,
with minimum J 0.94899389. Exact shell triangle tests at both initial
and terminal coordinates found zero self- and cross-layer intersections.
The initial point-in-bone count is a separate test. Other tissue pairs
and the shell between these captures are outside this audit.

All 34 registered ligament rows remained active, the terminal ligament
and fascia audit passed, and all 64 mechanical and vascular residuals
stayed within their existing thresholds. Pleural pressure was enabled
and ranged from -490.332520 to -490.332489 Pa during this short release.
Three physical skin/bed contacts supplied 0.38728 N on the final root
against 952.86 N of body weight.

The body stage took 156.953 s, or 0.00000637×. The whole process took
173.948 s, or 0.00000575×, without viewer presentation. The 64-root
comparison uses the same timeline and acceptance settings as e8ac916.
Krylov columns fell from 650 to 558. The body-stage elapsed time
changed by less than 1%. Supported equilibrium, clearance through a
breath, the 16-second run and 0.1× remain unqualified.

The reported shell is the reduced mechanical model. Its original
high-resolution source repair and static quality checks are retained
separately and have no continuous motion qualification here.
