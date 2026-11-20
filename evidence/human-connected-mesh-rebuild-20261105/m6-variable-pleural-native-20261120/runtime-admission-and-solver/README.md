# Matter runtime admission and full-volume Newton continuation

Matter initialization previously reported success before checking late
command-buffer invariants. A rejected runtime was never installed but
returned an encoded status, causing Human to read a zero timestep and
report an unrelated respiratory configuration failure. Both pinned
runtime owners now report success after installing the candidate.
Physical tolerances, commands, forces and states are unchanged by
this initialization patch.

A two-compartment Metal fixture reproduced the false success on Apple
M6. It now reports the actual flat-column incompatibility with an
uninstalled state. The compatible fixture remains initialized at
1 ms. The zero-context patches apply to the two source overlays in the sibling
pleural–abdominal-cavity evidence. The build and regression fixture
use the existing native owners.

The full-volume native diagnostic 012 restored the numerical settings
of the accepted matched full-coordinate 335 control, including the
reference-clearance treatment and compensated vascular state. It
retained the repaired geometry, exact mass mapping, ligaments,
deformable bed and variable pleural-pressure mode. The first Newton
direction was clipped to 0.817489 by continuous contact. Subsequent
directions used a full fraction and vascular backtracking did not
reduce them. Four passes reduced the fresh mechanical equilibrium
residual to 0.00744380. The vascular residual was 2.13e-8, below
its 1e-5 gate. The four-pass diagnostic still rejected on the
mechanical certificate and published no time.

The instrumented median CPU-arrival intervals were 0.252 s for
assembly, 0.786 s for the linear solve and 0.075 s for globalization.
The final certificate interval was 0.339 s. These are timings of
the tested solver settings for this root; integrated real-time
performance remains unqualified.

## First accepted full-volume steps

Run 013 used the normal 16-pass Newton budget and the admission fix.
It accepted both requested roots on the Apple M6. Accepted simulated
time was 31.25 microseconds. The coupled body and physiology clocks
matched, the trace reported zero mass residual, and the existing
bed owner received no measurable load at this early point. Root
assistance was disabled. Fresh mechanical residuals were 4.57e-6
and 3.40e-6 against the existing 1.25e-5 threshold. Vascular
residuals were 7.21e-9 and 8.99e-9 against 1e-5. Both steps were
physical accepts. Pleural pressure was -490.33249 Pa at both samples,
within the resolution of this abbreviated interval.

The integrated physics interval was 35.039 s (8.92e-7 times real
time), and the total invocation took 176.698 s (1.77e-7 times real
time). Presentation was disabled for this test. The run did not
settle onto the bed or qualify whole-body crossing counts in motion.

## Extended full-volume replay

Run 014 attempted 16 roots with the viewer disabled and accepted the
first 12. It completed 187.5 microseconds of coupled physical time.
The invocation took 354.398 seconds, including preparation and
the rejected root (5.29e-7 times real time). Every accepted trace
reported zero mass residual and zero root assistance. Pleural
pressure remained -490.33249 Pa at the emitted precision over
this interval.

On the thirteenth attempt the right ventricular row (zero-based
vascular row 16) failed its original normalized residual threshold.
The residual was 1.35619e-5 and its threshold was 1.0e-5. The same
attempt exhausted all 16 Newton solves and all 1,024 Krylov columns.
Fresh mechanical residual telemetry was 4.95326e-4. The accepted
body and respiratory indices and the Brain timestamp remained at
step 12. The rejected candidate was never presented or published.
This is an accepted microsecond-scale release prefix, with the
settled bed, zero motion crossings and 0.1× targets still open.

## Hydraulic predictor comparison

Run 016 used the existing compensated hydraulic predictor and its
compatible direct solver scheduling. It retained the same timestep,
mesh, mass, loads, contact, and certificate thresholds as 014. It
accepted the same 12 coupled steps and rejected the thirteenth. Its
total invocation took 380.595 seconds (4.93e-7 times real time).
The first ten accepted step submissions took 132.318 s in 016 versus
109.851 s in 014. The runs also differ in presentation and command
scheduling, so this is a comparison of the tested configurations.

The thirteenth step's final vascular domain norm was 1.58e-8, below
the 1e-5 tolerance. The final mechanical norm was 5.32407e-4 and the
step exhausted all 1,024 Krylov columns. On Newton passes 0 and 15,
the constitutive safeguard for object 15 (registered skin/fat)
restricted the correction to 0.125 and 2.38419e-7. Its minimum
volume ratio on the last such trial was 0.350000024. The last
coupled fraction was 5.06089e-8 before the vascular line search and
was unchanged by that search. The recorded mechanical certificate
failure, on object 13, and the constitutive limiting object are
different witnesses. The predictor configuration was not promoted.

## Exact native surface endpoint

Run 017 repeated the accepted 12-step prefix without the predictor,
this time with native presentation and a terminal geometry capture.
The body advanced 0.000187500009 s over 162.905 s of integrated
body runtime (1.151e-6 times real time). The complete 304.517 s
invocation ran at 6.157e-7 times real time. All 12 steps reported
zero mass residual and zero root assistance. The last step carried
zero measurable bed load, so the capture is early release motion.
Variable pleural-pressure mode was enabled. The emitted pleural
sample stayed at -490.33249 Pa over the completed interval.

The final native surface and fascia certificates passed. The fascia
audit recorded 26 regions and 34 ligament rows over 12 accepted
steps, zero ligament force residual and maximum tension 0.033782 N.
The independent exact surface check used the authenticated terminal
pack and its 157 body poses. It found zero skin self-pairs and zero
skin-to-bone triangle pairs. Other skin-to-structure triangle pairs
totalled 14,536 across 13,044 muscle, 76 tendon, 624 organ, and 792
vessel pairs. The organ total includes 350 pairs against ocular source
surfaces. Counts are raw witness pairs. They require anatomical
classification, and do not establish the number of distinct defects.
No contained pairs or internal-interior combinations were measured.
The complete report and the native accepted geometry receipt are
retained in this directory. This single endpoint makes no claim about
continuous motion clearance, supported equilibrium or 0.1× performance.
