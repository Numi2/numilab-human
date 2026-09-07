# Myofascia horizon qualification

The current Human/Matter stack passes four 10-microsecond steps, but fails
both eight and sixteen steps in a fresh Apple M4 Pro build. This is an open
contact-certification boundary, not sustained tissue or standing validation.

The runtime is clean revision `3561113777afddbbf2e37965afaddad983772282`.
All eight input payload hashes match the published directional-myofascia
receipt. The missing tendon payload was regenerated from current importer
source and matched `72ca0ec4ef53f647784a89f761e78495d4997c7d72cfe6a1d9eedb24dd97feaa`.
This experiment explicitly excites source muscles 186 and 198 by 0.02 above
the compiled posture baseline; it does not infer the old experiment's
unrecorded invocation from its output.

| Horizon | Simulated time | Result |
| --- | ---: | --- |
| 4 steps | 0.04 ms | accepted, bitwise replay, verified rollback, all four reaction audits nonzero |
| 8 steps | 0.08 ms | rejected, Matter contact failure, zero minimum reaction audit |
| 16 steps | 0.16 ms | rejected at the same reported contact boundary |
| 64 steps | 0.64 ms | not run after the failed 16-step gate |

The four-step result reports 0.06227 mm maximum displacement, minimum
`J = 0.991260`, and 3.66308 N minimum reaction L1. The coupled transaction took
5.440 seconds (0.735 steps/s); the complete process took 37.806 seconds and
peaked at 376,209,408 resident bytes. This timing includes the native coupled
transaction, not an isolated FEM kernel. Retained native allocation and GPU
counter profiling were not measured. The front runtime frame was inspected;
this experiment does not add a multi-angle visual qualification.

Both failed horizons report `accepted_status=6`, object 0, index 1040, and
contact identifiers 177 and 487. `NM_STATUS_CONTACT_FAILURE` is defined in
Matter's `shared.h`. The probe's generic error message labels diagnostic x as
`accepted_minimum_J`, but contact failure uses that field for a primitive
identifier. **177 is not a measured deformation Jacobian.** The error does
not identify the first failing step. Only the four-to-eight-step interval is
established; no narrower failure time or anatomical pair is claimed.

The next mechanics work is to identify that compiled contact pair, check its
source geometry and eligibility, and close contact certification without
turning off inter-object contact or relaxing admission solely to pass this
probe. The existing objects already disable same-object self-contact. No
material, contact threshold, force-owner fraction, or geometry was changed
in this qualification increment.

## Repeatable command

Build `metalrobo_numilab_human_myosim_visual_probe` from the intended runtime
revision using CMake Release on the Mac mini. Place the eight named payloads
listed in `qualification.PAYLOADS` together in one input directory. Then run:

```bash
PYTHONPATH=src python3 -m numilab_human.cli numi-human-myofascia-qualify \
  --runtime-root /path/to/numi-lab \
  --runtime-build /path/to/build \
  --input /path/to/paired-payloads \
  --output /path/to/new-receipt-directory \
  --steps 4 16 64
```

The command binds the runtime root to CMake's build owner, records input,
binary, library, shader, runner and build-cache hashes, retains exact arguments
and stdout/stderr, and atomically updates its receipt between horizons. Each
horizon has a bounded process-group timeout. Existing evidence directories
are never overwritten. A failed or incomplete native result returns exit 1
and prevents later horizons. Peak resident memory is process memory, not
Metal retained allocation; unavailable measurements stay null.

Admission requires completed Human and fascia horizons, all-step reactions,
832 endpoint transfers per step, nonduplicated same-command-buffer ownership,
positive finite deformation evidence, Apple device agreement, bitwise replay,
and verified rollback. It does not require or imply `compiled_stand_balanced`.

Evidence is retained under [media/myofascia-horizon-20260907](media/myofascia-horizon-20260907).
The full importer/qualification suite ran on the Mac mini: 113 tests,
107 passed and six skipped because their optional fixtures were unavailable.
The final parser also admits the retained real four-step transcript and
rejects the failed native transcripts.
