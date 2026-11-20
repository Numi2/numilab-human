# Full-scene equilibrium recovery on Apple M6

The published refit and mass transfer are the inputs to both trials.
The sole physical executable difference in trial 023 is the Metal
sweep patch in this directory. It adds two bounded dyadic prefixes
to the existing conservative polynomial contact check. A positive
result still proves strict separation at the original immutable
clearance for every time in the returned interval. The existing
merit, contact, equilibrium, and rollback checks still decide
whether a step can be committed.

## Captured contact

Trial 022 reproduced the failed thirteenth step and recorded the
four compensated endpoints of contact 2222094. The vertex and
triangle both belong to the registered deep fat boundary. The
vertex is 15.02 nm from the nearest triangle corner in the stored
rest coordinates. It is incident to only one tetrahedron and the
contact triangle belongs to a separate tetrahedron that shares an
edge with it.

At Newton iteration 15, the captured proposed motion closes the
gap from approximately 1.0101 nm to 0.8153 nm. The independent
reconstruction places the 1 nm margin crossing around 0.051876 of
the correction. The pinned production Metal probe certified
0.00202934; the patched production Metal probe certified 0.03125.
The unchanged feature certified a full step in both builds. The
deliberate crossing control remained clipped in both builds.
The proof uses the recorded compensated relative-frame positions
and unchanged margin. The independent floating-point reconstruction
locates the witness, while the GPU interval proof certifies the
accepted prefix.

## Coupled transaction

Trial 023 accepted all 16 requested roots at 15.625 microseconds
per root. The preceding 12 accepted physiological/body CSV rows
match trial 021 exactly, and both used 2,224 active Krylov columns
over those 12 roots. At root 12, the old run exhausted 1,024 columns
and measured a mechanical residual of 5.63911e-4. The patched run
used 384 columns and passed the same 1.25e-5 gate with residual
7.49081e-6. Its vascular residual was 1.07319e-7 against the
unchanged 1e-5 gate. The four additional steps were accepted
within the existing coupled body and physiology transaction.

The terminal fascia audit passed with 26 anatomical regions and 34
registered ligament rows. Ligament force residual was zero at the
reported precision. Body mass residual and root assistance were
zero in the accepted trace.

The run advanced 0.250000012 ms in 394.765 s including startup
(6.33288e-7×). The native body timer measured 195.665 s
(1.27769e-6×). The matched prefix took the same number of Krylov
columns, so the advance here is recovery of a rejected step and
expenditure on that step rather than an established steady-state
throughput gain. Presentation was disabled.

Variable pleural-pressure coupling was enabled. The accepted
pressure samples all display −490.332489 Pa at this sub-millisecond
horizon. The terminal bed load was still zero. This test qualifies
the short release prefix and its equilibrium acceptance. It leaves
the supported 16-second motion at 0.1× and the independent
whole-body crossing audit open.

## Replay

The launch receipts pin the native executable, linked runtime,
Metal libraries and input plans on SSH Numi. Build the single
Metal patch against the trial 022 source using the supplied compiler
command. Build the existing-probe driver with clang++, Foundation,
Metal and ARC. Run check_prefix.py on the captured witness with
the old and patched libraries. Run run_matched_native_trial.py on
the published trial 021 launch directory with the patched library
to reproduce trial 023.
