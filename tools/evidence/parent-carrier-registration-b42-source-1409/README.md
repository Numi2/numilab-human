# Existing parent carrier registration preparation

All eight current source004 NHTISS parent carriers were re-emitted by the existing carrier owner for registration 31c77660 after both registrations were checked against the consumed NHRIGID. Atlas-coordinate carriers changed 84 binding records across source slots 0, 1 and 5. Core-local carrier identity transforms (slots 4, 6 and 7) remain byte-exact; unchanged back bindings also remain exact.

Every geometry/normal/weight/index/row byte is preserved, along with the existing source term order and Float64 coefficients. Child cf7d35ae is bound in the prepared extension. The full extension remains on SSH Numi with hash 994514d561c7bea8e1e4c63168c3fbf6d3bbe93dbfb5882da855a18022ca1f0e.

Attempt001 produced all eight payloads but its report finalizer incorrectly treated the provenance dictionary as a list. That failure is retained. The verification successor compares the already emitted bytes to the exact owner result without repeating emission.

Focused carrier checks: 12 tests and 14 subtests passed; Ruff passed. This is source preparation only; integrated receipt composition and native child-rest reconstruction remain pending.
