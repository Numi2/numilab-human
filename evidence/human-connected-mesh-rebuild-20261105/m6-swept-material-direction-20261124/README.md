# Complete human, second physical root on Apple M6

Both runs accepted one 7.8125 µs root with variable pleural pressure. The three accepted traces (physiology, support impulse and momentum) match byte for byte. Recompiling the unchanged physics shader reproduced the original Metal library byte for byte.

At the final Newton iteration in the original solver, fat element 1197378 in the right-ulna region reached its upper material volume ratio of 2.5. The original direction builder evaluated the full endpoint. Material backtracking on shorter trial steps took 29 trials and limited the fraction to 3.73e-9. Deformable contact reduced that fraction again. The final applied fraction was 8.088681e-10.

The changed sparse direction builder evaluates extrema of the cubic determinant along the proposed segment. When the segment crosses a material bound it corrects the appropriate determinant control using its analytic derivative. The same independent material, contact and coupled residual checks still govern the native trial. A CPU reference on the M6 checked 4,427 valid-endpoint segments, including 479 interior violations, and the directional derivatives of all three controls.

The changed solver's final material trial admitted the full endpoint. The solver applied 0.9000141 of the proposed correction after downstream checks. The fresh mechanical residual fell from 0.0362009034 to 0.00152729778. Active Krylov columns fell from 560 to 393. Relative correction increased from 1.89235e-08 to 0.0281079. Both attempts rejected root two and preserved the accepted history. Neither result establishes sustained throughput.

The matched GPU attribution reproduced the final residual to relative error 1.44e-08. The largest residual span was registered bonded skin and fat. Its fraction of squared residual was 0.99999017. The peak skin/fat node was 140573 with donor body 92. The remaining nonlinear imbalance and live posterior mass/force transfer still need resolution. This experimental solver has not qualified the delivery scene.

At the largest residual node the material and bed direction operators were examined directly in the final Newton iteration. The registered left-ulna neighborhood includes 22 cells and 14 nodes. The RMS velocity-direction changes over those nodes were 0.20006739 m/s after material feasibility and 0 m/s after bed feasibility. The accepted traces and terminal failure diagnostics reproduced the paired trial. The measurement covers those two transformations of the local search vector. It does not isolate every force term in the nonlinear residual.
