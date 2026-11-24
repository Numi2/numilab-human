# Posterior tissue forces in the native Human scene

The native Apple GPU experiment joins 51 registered deep-fat nodes to muscle material parents with reactions on the pelvis and both femurs. The same runtime applies both sides before generalized-load projection. The reference operator uses 2,256 tetrahedral integration cells and the existing passive muscle compliance. It is a corotational energy with a condensed reference tangent. The original donor inertias still carry 42.86 g of reconstructed fat. Large-strain material response and independent tissue inertia remain outside this approximation.

## Matched physical result

| Measure | Result on SSH Numi |
| --- | --- |
| Accepted time | Two 7.8125 µs steps, 15.625 µs total |
| Body speed, including the coupled body clock | 0.000000221× |
| Cold process | 214.47 s |
| Normal bed force and normal weight | 11.015 N / 952.862 N |
| Third step equilibrium residual | 0.03183367, versus 0.03182468 in the control |
| Third-step Krylov columns | 552, versus 552 in the control |
| Material surfaces at step two | 0 intersecting pairs across the four material walls |
| Skin and fat | 0 inverted cells, 4 skin and 34 fat cells below the shape threshold |
| Actual skin and fat intersection with 185 cortical banks | 5.9764 cm³ |
| Largest change in accepted skin/fat node position against control | 0.0399 µm |

The passing short run used the full shared tissue solver, the fixed bed and variable external vascular pressure. The two GPU force observations balance to within 3.11e-11 N. The terminal net tissue reaction was about 0.00017 N in magnitude. The failed third step preserved two accepted steps. The results include separate physiological and mechanical clocks. Native presentation and sustained performance are unqualified.

## Where the load goes

In the earlier patch, fixed cut nodes carried almost all of an imposed bed load. Here the cut is traction-free. A unit perturbation at the two currently loaded sites in the registered nine-site gluteal footprint gives 3.760 mm/N of work-conjugate compliance with the three skeletal donors held. All 2,814 material nodes are free. The full measured footprint gives -2.081 mm/N on the opening prestressed geometry. This negative direction rejects that local equilibrium approximation.

Skin vertex 24331 carried 66.98% of the terminal microstep normal impulse. Its eight neighboring surface edges span 6.2–10.0 cm. The source maps the nearest right infraspinatus to the humerus and scapula. At this vertex the bed-normal ray traverses about 0.025 mm of infraspinatus and first reaches scapula about 29.6 mm below the skin. Neighboring rays traverse several different tissues. The accepted vertex is 20.6 nm above the registered plane at the endpoint. Terminal impulses refer to the preceding microstep. The clipped contact area at the endpoint is zero. A frozen 1 mm plane advance gives 138.1 mm² over the same eight triangles.

Treat the initial shoulder pressure as a localized transient. The next physical registration needs the distributed skin, fat, shoulder-muscle and cortical interfaces under an actual gravity preload. A single vertical spring at the peak node would infer a load route across several tissues.

## Reproduction

The verified archive contains the exact two-step and three-step launches, runtime modules, generated sources, native captures, physical state, independent tests, trial receipts and the earlier rejected local witnesses. The build script reuses the pinned Metal pipeline and existing Human adapter. The opt-in operator fingerprint covers its reference geometry, donors, force law and matrix. The runtime verifies the matrix digest at load. The normal production model remains the control.

The independent small-deformation and combined-rotation tests directly invoke the compiled Metal kernel and compare all 51 scattered tissue forces and all three skeletal wrenches against virtual work. The largest relative error in these two loaded tests is 6.31e-06. Rigid motion tests use absolute errors near the zero-force state. The clipped-face test conserves projected area and first moment. An initial three-step command requested a capture at step two with incompatible submission cadence. It exited during setup. The retained physical comparison uses the separate, matched three-step launch.
