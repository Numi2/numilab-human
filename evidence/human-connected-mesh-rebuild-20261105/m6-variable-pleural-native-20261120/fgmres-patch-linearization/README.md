# Cached tetrahedron patch preconditioning on Apple M6

The native full-tissue profile recorded 240 Krylov columns at root 1.
Each column reconstructed the four patch contributions from the same
inverse diagonal and reread active element flags during node gather.

The optional direct Metal path compacts the active component masks, local
node indices and clamped diagonal sum into a 32-byte tetrahedron record.
It regenerates the record at every restart, after the inverse diagonal is
available. The current correction still reads each column's actual basis
and uses the same ordered FP32 sum and division. The gather uses the same
ordered CSR incidences and patch count, with the compiled activity marker
in the cache. Completed environments leave the cache and scratch untouched.
The end-of-step physical certificate still runs in the original owner.
The optional prefix ICB path retains its original kernels.

The paired control uses the same rebuilt executable and Metal library
with the cache disabled. Both runs keep the mass data, contact policy,
constitutive law, active-set and numerical tolerances pinned.
Instrumentation samples root 1, holding the stage boundaries fixed.
This headless measurement covers only an accepted three-root prefix.
Physiology is active and the emitted pleural pressure is constant within
that short window. The input scene remains the full registered volume.

The Apple M6 fixture compares the old production kernels with the cached
kernels byte for byte across mixed coordinate masks, inactive elements,
masked NaNs, a guarded invalid corner, and reactivated environments with
changed diagonals. The native gate compares every emitted physical CSV
and checks that the Krylov work counts match the control.

## Reproduce

Use the pinned source and libraries on SSH Numi:

```sh
python3 prepare_patch.py
python3 run_profile.py cached-root-1
python3 run_profile.py --cache-off control-root-1
python3 ../muscle-shell-registration-018/check_profile_matched_roots.py control-root-1
python3 check_comparison.py control-root-1
```

The check fixture builds with the retained Matter include directory and
`clang++ -std=c++23 -O2 -fobjc-arc -framework Metal -framework Foundation`.
Pass the original and candidate `NumiMatter.metallib` files to the fixture.

## Matched device result

| Measurement | Cache off | Cache on |
| --- | ---: | ---: |
| Root 1 linear GPU time | 9927.388 ms | 9385.508 ms |
| Full invocation, three roots | 188.894 s | 187.070 s |

The root 1 sample used 240 Krylov columns in both
runs. The measured reduction in linear-stage GPU time was
+5.46%. The cache used 47,207,136 GPU bytes.
All 10 emitted native CSV files match byte for
byte. This invocation covers a headless, 46.875-microsecond release prefix.
Supported motion, continuous crossing checks, physiological response and
steady-state throughput remain open. The cache is opt-in.
