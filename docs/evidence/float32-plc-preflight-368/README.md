# Fail early on collapsed emitted boundary facets

The native-domain adapter now measures areas from the exact emitted Float32 coordinate values before hole search, PLC generation, TetGen or parent mass work. A collapsed facet produces the existing adapter receipt with deterministic face-row witnesses and no mesher invocation. Positive submicron facets are allowed by this gate: separate construction quality and volume admission remain authoritative.

Three targeted regression cases cover rounding collapse, distinct collinear points, and a positive submicron facet. A CLI regression verifies that rejected geometry reaches neither capacity/hole search nor TetGen. The existing native-domain and role-local surface-map suites also pass. On the current O026/I-fill003/D366 boundaries, the new preflight reproduces exactly 772 invalid I faces in about 19 ms; O and D have zero collapsed faces. This is an early rejection instrument, not a physical qualification.

The core366 archive was independently downloaded from GitHub. All thirteen manifest payload files match, and restored geometry-subset/source-ancestry replay passes. The original exact-predicate audit was not rerun by that subset replay.
