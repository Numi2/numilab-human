# Clear the posterior muscle side of the fat interface

Question. Can the posterior gluteal contact surfaces be fitted outside the
exact cooked skin/fat boundaries without moving the shell or the skeletal
attachment neighborhoods? Previous fits changed the shell side and retained
250 gluteal-to-layer pairs in the authoring input. Recount on the current
native cooked shell before treating this as the new baseline.

Use both registered gluteal volumes, the native cooked shell, the native
bone triangle inventory and the previously captured soft tissue and FEM
surfaces. Preserve each gluteal volume, original cell and source face
identity, and every gluteal vertex on an existing bone crossing face. The
bone freeze is an authoring constraint. It cannot create a runtime force.

In one bounded run per muscle, allow motion along the bed normal within
7.5 mm for nodes within two incident-tetrahedron rings of exact witnesses.
Require cell determinant ratios of at least 0.55 in the linear program.
After Float32 emission check positivity, the existing normalized 1e-5
shape floor, relative volume closure within 1e-6, unchanged frozen nodes,
zero fat/skin pairs, zero self-intersections and no newly intersecting
other bone, muscle or continuum faces. Stop after four active-set rounds.
The unchanged emitted geometry and its exact pairs are the control.

A passing candidate is a local surface repair. Source bone intersections
at muscle insertion and neighboring anatomy have separate witnesses.
The native coupled test also requires signed containment, anatomical
attachment review, typed mass repartition and load transfer. This run
cannot certify those outcomes by analysis of the isolated volumes alone.
