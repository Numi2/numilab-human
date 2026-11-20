# Full-tissue native Krylov convergence

SSH Numi ran the full 22-object continuum scene with the repaired skin
and fat, registered ligaments, active physiology and variable-pleural
mode. All three roots accepted in both observation runs. All ten native
CSV files match the previous cache-enabled control bit for bit. The
complete prefix advanced 0.046875 ms, spanning no measurable pleural
pressure change. The initial pack was recovered from its existing
verified compressed copy before retiring each duplicate raw file.

The selected second root used 240 Krylov columns in 15 cycles. All four
active Newton corrections used `alpha=1` at every captured limit and
the final line-search boundary. The separate residual Armijo observer
was unavailable because that residual search policy was inactive in
this configuration. True linear defects below come from the freshly
evaluated coupled physical operator. The row-scaled Arnoldi norm and
unscaled physical impulse norm use different metrics.

| Newton pass | Physical RHS norm | Last physical linear defect | Reassembled nonlinear norm |
| --- | ---: | ---: | ---: |
| 0 | 1.472e-4 | 2.099e-8 | 2.090e-5 |
| 1 | 2.090e-5 | 2.055e-8 | 1.821e-5 |
| 2 | 1.821e-5 | 3.719e-9 | 1.176e-5 |
| 3 | 1.176e-5 | 8.464e-7 | 3.990e-6 |

The fourth Newton pass skipped its first cycle and used 48 columns in
three subsequent cycles. The last two cycles reduced the physical
defect by 2.12% after an earlier reduction from 1.176e-5 to 8.648e-7.
Their projected residual ratios were 0.99285 and 0.99940 relative to
their restart reference norms. In the second Newton pass, the final
restart raised the physical linear defect from 1.123e-8 to 2.055e-8.
The physical acceptance checks still certified the whole nonlinear
step. Contact globalization imposed no step reduction on this root.
These observations support testing a more effective preconditioner or a
restart strategy driven by the actual physical residual on a broader
root sample.

The bed must still balance approximately 953 N of weight. The full
volume initializes in release mode. All 76,153 admitted outer-skin bed
pairs have zero measured load on the terminal short step. The retained
longer 16-root release also had zero terminal support, 14,179 raw
nonocular skin-to-tissue triangle pairs, zero skin-to-bone and zero
outer-skin self-pairs. No supported equilibrium, sustained throughput
or 16-second zero-crossing motion certificate is established here.

## Bounded Arnoldi row transforms

The opt-in runtime overlay dispatches the in-place scaling over only the
mechanical node domain. Out-of-place unscaling uses the same GPU active
range and copies the complete identity-scaled tail with a node stride.
Current Arnoldi activity selects both ranges. The begin/restart RHS,
active vascular and rigid rows, physical operator, fresh certificate
and line-search decisions keep their original owners. The complete
Krylov storage layout remains allocated.

The standalone Metal fixture passed bounded and direct transforms
bitwise, including nonaligned counts, identity rows spanning several
strides and an empty indirect dispatch. The same binary and assets were
then run with the option on and off. Both accepted three roots with
identical iteration, residual and accepted-state telemetry. Ten native
CSV files match bit for bit. Root 1 linear GPU time fell from 9,433.29
to 9,164.97 ms (2.84%). The measured accepted interval improved from
1.3814e-6 to 1.4150e-6 times real time. Whole process wall time was
186.41 s off and 186.92 s on. These are one-pair headless observations,
with no sustained or native-viewer speedup claim. The start-up wall
measurement includes initial cook and import. The smaller GPU stage
result alone is insufficient to establish an end-to-end speedup.

Commands on SSH Numi from this case directory:

```sh
python3 run_native.py --column-trace root-1-column-trace
python3 run_native.py --column-trace --physical-trace root-1-physical-trace
python3 prepare_row_transform.py
./run_row_transform_pair.sh
python3 compare_row_transform.py row-transform-on row-transform-off
```
