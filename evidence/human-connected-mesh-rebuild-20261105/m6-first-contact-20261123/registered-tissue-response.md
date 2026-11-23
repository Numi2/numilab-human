# Registered tissue under the fixed bed

The detailed volume contains 705,090 dermal and 743,709 fat cells. The native mass mapping assigns them 3.452571283 kg and 10.487108427 kg. All 14 deep anatomical attachments and all 19 outer nodes in the first millimetre above the release bed share one connected volume. Another 88 fat components have 0.377 mg of mass between them.

The 16-step native first-contact endpoint had eight failed material cells, all on the main component. Twenty-six of its 29 deep-fat triangle crossings also occurred entirely on that component. The other three occurred within component 47. Removing small islands would leave the main failure intact.

The continuous registration on recorded 16-second skeletal motion reduced 21,882 inverted cells to 237, with all eight tested finger extensor insertions inside. All 237 inverted cells belong to the main component. Applying a unilateral normal contact constraint removed a 6.56 mm penetration of the bed but raised the inverted count to 253. Both registration proposals failed admission.

For the reference pose, the contact calculation uses the installed 3D material tangent, each cell's routed mass and the registered attachment rows. It solves on an expanding patch, then evaluates the residual on every incident exterior row. The full mass matrix supplies a lower bound on the shifted positive elastic operator. The resulting energy inequality bounds the compliance error for any combination of the 19 normal loads. A separate solve with an additional layer of nodes tested the bound at every step size.

| Step | Active material nodes | Worst-case relative compliance bound | Measured change on larger patch |
|---|---:|---:|---:|
| 15.625 µs | 27 | 0.1830% | 0.1752% |
| 125 µs | 187 | 0.0670% | 0.00683% |
| 0.5 ms | 471 | 0.1179% | 0.00367% |
| 1 ms | 673 | 0.4771% | 0.00333% |
| 2 ms | 4,643 | 0.5685% | 0.000424% |

These limits cover initial skin/fat compliance while the deep attachment positions are fixed. Settling changes the footprint and the tangent. Neither a physical timestep nor end-to-end throughput can be qualified from this calculation.

On SSH Numi, the installed first-contact tree supplies the material reader in its preparation environment. Compile the small Accelerate helper in the output directory and run the module against the matched registration checkpoint. Run the regression from that same output directory.

```sh
clang++ -std=c++17 -O2 -dynamiclib "$module/factor.cpp" -framework Accelerate -o material_factor.dylib
"$PY" "$module/adaptive_contact.py" "$registration"
"$PY" "$module/test_adaptive_contact.py"
```

[Source identities and measurements](registered-tissue-response.json) · [Verified GitHub recovery](registered-tissue-response-recovery.json)
