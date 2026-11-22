# External balance and a localized contact failure on Apple M6

The loaded adult weighs 952.86444 N. A cold native initialization summed the rebased articulated COM poses and all 318,226 compiled FEM nodes in 22 objects. Their mass is 97.13195038 kg and their combined center is (−0.02518770, −0.77041893, 0.10327048) m in the bed frame. The skin and fat share that mass partition.

We integrated bed pressure over the registered exterior skin triangles in the existing Matter residual and FGMRES operator. During release the stiffness field is interpolated from the previously accepted material nodes. The first implementation failed its approaching-skin derivative comparison by 3.83%. The corrected M6 kernel passed with relative errors of 0.000159 for stationary skin and 0.000221 for approaching skin. Neighboring nodes receive the normal load and its off-diagonal stiffness. The fixed bed supplies the sole normal reaction.

The original loaded geometry supplied 331.26833 N under this contact law. We fitted the posterior displacement using the actual global vertical load and its two tipping moments. The largest coordinate change was 2.001 µm. The production force kernel at the emitted coordinates produced 952.86415 N. The two moment residuals were −0.00000973 and −0.00000664 N·m. The whole-volume check passed 1,448,799 cells for geometry, stretch and stress. The maximum donor first-moment change was below 8×10⁻¹⁴ kg·m. The active band contains 138 exterior nodes and 208 nodes receive distributed load at initialization. The native runs used variable pleural pressure.

## Free release

The mode-off native control reproduced the prior two accepted force histories and the same third-step physical residual of 0.03632484. The fitted area-contact candidate rejected at step zero with mechanical residual 0.01603335. A signed-normal plane limiter also rejected at step zero with residual 0.01603508. The acceptance threshold remained 0.0000125. The alternative limiter passed 16,384 affine sweep checks. The existing motion baseline remains the best demonstrated dense release.

The full scene's limiting registered skin node was 24013. Its four incident faces span neighboring loaded heights from 1.2 to 14.0 mm. The starting vertical elastic force on its 1.116 g of tissue was −125.52 N. At the last Newton iterate the node sat at 1.00975 nm above the plane and its correction requested 14.14 µm further inward. Its upward bed reaction was 0.03133 N. In an independent sensitivity check with neighboring vertices held at their initial coordinates, increasing the integration order from 4 to 256 changed the reaction from 0.03101 to 0.03194 N. The initial elastic force and the limiting bed reaction refer to distinct states.

A second fit included the local elastic and gravitational forces of the complete posterior cell stars. It reached its iteration budget with 0.199 N vertical imbalance and failed the prescribed force and torque checks. The next physical issue is unilateral support at a corner where contacting area tends toward zero, together with the initial internal tissue stress. A settled-body claim requires an accepted release and measured load transfer through motion.

The dense baseline's last qualified geometry capture is at 31.25 µs. It passes the skin/fat cell gates and contains 10,895 exterior skin intersections with muscles, tendons and vessels. The new fit has yet to qualify through 16 seconds of motion or establish sustained throughput.

## Reproduction

The recovery receipt identifies the independently downloaded private Human GitHub release archive and SHA-256 for each member. It contains the original and corrected contact source, native executables, selected dylibs and metallibs, exact launches, force observations, failed roots, fitted coordinates and parent recovery receipts. The full-scene trials used the existing run_cavity_motion.py entry point with the registered resting adult on SSH Numi. The checks and launch receipts identify the native executable and both libraries selected for each run.
