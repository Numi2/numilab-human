# First accepted material-supported step with the registered ligaments and fascia

The regional skin/fat bed owner now admits an explicitly authored compressed release after checking every one of the 61,950 posed skin vertices against its material section's available thickness. The body is translated and rotated at construction. The bed, its 32 material areas and force law continue through the native contact pipeline. Both cortical material endpoints of all six pose-dependent shoulder fibers were certified again on the emitted skeleton. All 185 source bone banks clear the fixed plane with a minimum clearance of 18.27 mm. The regional mass ledger closes to 0.84 mg over 97.13 kg.

The earlier fully registered release put zero impulse through the bed. The new starting pose carries 961.27 N before dynamics under the original section laws. The first accepted Apple M6 step transmits 960.95 N of normal force, calculated from the sum of the six positive impulses divided by the accepted step duration. The full-body gravitational load along the normal is 952.86 N. The net tangential response is 283.32 N. The source-region impulses close against the independent world momentum export. Both runs produced byte-identical impulse and momentum records for their first step. The limb loads still require equilibration.

| Run | Accepted time | Cold wall time | Outcome |
| --- | ---: | ---: | --- |
| Single step at 0.125 ms | 0.125 ms | 17.34 s | Normal exit |
| Eight steps requested at 0.125 ms | 0.125 ms | 19.24 s | Fresh passive tissue certificate rejected step two |

The initial generalized acceleration residual increased from 12.89 to 119.33. Maximum local compression reached 61.52% of inferred skin/fat thickness, and the physical fidelity of the linear column law at this strain remains unqualified. The new pose is an experimental checkpoint. It cannot be used as evidence of equilibrium, sustained motion or absence of skin–organ intersections. The denser continuous skin/fat FEM remains a separate model.

In the rejected second root the first Newton update reduced the linear right-hand-side norm from about 0.00035 to 0.000026. Subsequent candidates increased it repeatedly. On the final linear cycle the directly recomputed residual was 1.66e-8 and the independently assembled nonlinear residual for the candidate was 1.212e-4. These describe different equations. The physical certificate retained the last accepted state. This result focuses the next solver investigation on nonlinear update acceptance and the implicated tissue rows. Extra linear iterations alone do not establish equilibrium.

On SSH Numi the reproducible short run is

```sh
cd ~/Work/numi-human-supported-shoulder-085/force-route-086
./run-supported-scene --launch connected-forearms-087/seated-native-certified-single-source/launch.json
```

The stopped run uses seated-native-certified-source/launch.json in the same directory. The two recovery receipts identify independently downloaded release archives with exact sources, binaries, shaders, traces and all retained failed preparations. The patch is relative to the registered plane owner at 72ccda65. Applying it reproduces all four compiled source hashes in source-identity.json. The previous release remains the control.
