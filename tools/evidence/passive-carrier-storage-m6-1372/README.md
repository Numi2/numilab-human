# Passive carrier Float32 storage regression

The existing NHTISS carrier encoder now rejects exactly zero-area triangles after conversion to the coordinates it will serialize. It uses the existing exact dyadic predicate; it does not impose a minimum triangle area, move geometry, or modify the runtime.

On ssh numi, the retained 412 candidate passes all six rows. The failed 424 interface candidate rejects rows 153–156 and accepts unaffected rows 157–158. The real-candidate test uses declared synthetic normals and a one-body field to isolate triangle storage; it is not anatomical or native qualification.

Unit coverage includes a nonzero Float64 triangle collapsed by storage, distinct collinear corners, and a nonzero subnormal triangle accepted without an area tolerance. Adjacent composition and carrier tests: 93 passed, 6 subtests passed. The initial unittest composition discovery found no tests; pytest was used for the function-based composition tests.

No new native run or five-minute qualification is claimed.
