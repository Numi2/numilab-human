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
