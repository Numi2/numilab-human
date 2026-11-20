# Exact contact query reuse in the coupled M6 scene

The 128-root matched replay uses the complete certified contact inventories
with the existing row cache, closed-bank AABB exclusion, exact motion-cache
invalidation and compacted GPU queries. It retains 413,997 physical source
rows and the same tissue, pressure, contact and solver tolerances. Contact
logic selects queries on the GPU. No contact loads, ancestry, masses or
force owners were changed for this performance comparison.

All 128 terminal native certificates, domain-residual diagnostics and Krylov
work records match the uncached control. Initial shell coordinates, startup
solid witnesses and the entire coupled CSV are byte-identical. The eight-root
endpoint comparison has zero regional force and strain failures across 91
regions. Body runtime drops from 252.440 to 98.599 s, a 60.94% decrease, for the
same 2.000 ms. The measured factor is 0.00002028× for the body and 0.00001732×
from launch. This is a headless native release replay. Its physical clock,
pleural-pressure variation and validations are active. Presentation was not
measured.

The 128-root release still carries 0.400701 N at three bed contacts against
952.864 N of weight. Root 111 still exceeds the separate tightened mechanical
domain target. Rigid plus FEM mass closes within 1.2 mg of the 97.13195 kg
inferred budget. The terminal shell has positive Jacobians and the 34 registered
ligament rows pass the terminal native audit. Exact shell checks find no self
or cross-layer intersections at the two captured endpoints. Outer and deep
point containment against 86 bone banks is clear at startup. These tests do
not establish whole-body crossing clearance between captures.

A separate four-root fixed-bed load-seed trial used the unchanged physical bed
and timestep, with a one-time 0.00435° root tilt and a positive 0.635 µm
alignment request. The existing barrier law predicted static reaction of
953 N if the three outer nodes were held in the emitted footprint. No such
holding force was applied. All four steps accepted and both captured shell
endpoints were clear. The three nodes moved above the 10 µm contact range.
Terminal bed reaction was zero. The lightest outer node moved from 0.646 µm
to 29.492 µm above the bed and had upward velocity of 0.352 m/s. This fails the
prespecified load-seed test. Supported equilibrium needs load on a distributed
contact patch and equilibrium under the coupled tissue, joint and pressure
laws. Replaying this initial tilt is not a route to qualification.

The full initial pack for that rejected seed was uploaded privately and
independently downloaded to verify its byte hash. APFS compressed copies of
raw logs and compact hashed run data are retained. The qualification requirement
remains 16 simulated seconds of coupled native motion at 0.1× or better with
variable pleural pressure and zero erroneous crossings.
