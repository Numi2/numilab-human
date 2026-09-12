# CVSim-21 source reference

This offline reference executes the original PhysioNet CVSim 1.0.0 C equations. It is separate from Matter's native runtime. Python only builds, launches, hashes and compares C/C++ outputs.

The unchanged files and individual download URLs are locked in [the source manifest](../../third_party/physionet/cvsim21/source-lock.json). Read the accompanying `NOTICE.txt`: PhysioNet designates ODC-By-1.0, whose text addresses database rights; the retained C files do not independently establish a model-level software license. This record does not relabel them as MIT, GPL or Apache.

## Two distinct source references

`headless.c` calls original `init_sim` and `step_sim` with compression factor 1, arterial baroreflex off, cardiopulmonary reflex off, and tilt off. It preserves `initial_ptr` / `mapping_ptr` / `estimate_ptr`, the discrete SA-node algorithm, source step control, and the source's disabled `fixvolume_ptr` correction. It exports every accepted original step, including the initial state: 21 pressures, 21 volumes, all 24 flows, external pressures, reflex state and cardiac timing. The original GUI export exposes only 21 of the 24 flows; this observer reads the complete original `Data_vector` directly.

`continuous_reference.cpp` uses the same unchanged `elastance_ptr` and `eqns_ptr`, identical initial estimates and fixed reflex values, with a prescribed period `60/70` seconds. It replaces the discrete SA-node timing algorithm with a continuously evaluated phase, then integrates 21 pressure states using an independent FP64 Dormand–Prince 5(4) solver. It does not call the native equations. This is a **continuous source-equation interpretation**, not numerical reproduction of the original discrete SA-node algorithm. `continuous-original-times.json` quantifies their discrepancy at the exact original accepted times, with no trace interpolation.

The regular reference output interval defaults to `float32(0.0005) = 0.0005000000237487257` seconds. Native 1 ms and 2 ms comparison times are exact subsets at stride 2 and 4. The final sample is the exact requested cycle endpoint and can be a shorter interval; it is not an extra native time step.

## Reproduce on an Apple CPU

```sh
python3 tools/cvsim21_reference/run_reference.py --output Build/cvsim21-reference
python3 tools/cvsim21_reference/analyze_reference.py Build/cvsim21-reference --output Build/cvsim21-reference-audit.json
```

The runner checks every source hash, builds with `-fno-fast-math -ffp-contract=off`, runs 100 original seconds and 20 prescribed cycles at tolerances `1e-10`, `1e-12`, `1e-13`, then runs the exact original-time comparison. It captures compiler, source, harness, binary and output hashes before/after execution. Run each original reflex configuration in a fresh process because upstream has static state. The headless executable exposes `ABReflexOn` and `CPReflexOn` independently as 0/1; the default qualification precursor exercises both off only.

## Source-specific boundaries

- Actual initialization is `init_sim -> initial_ptr + mapping_ptr -> estimate_ptr`; the legacy `initialization.c` is not used or vendored. Actual nominal blood volume is 5150 mL and heart rate 70/min.
- Absolute volumes follow source zero-pressure filling volumes plus distending storage. Nonlinear venous compartments use bounded `atan` laws. The parameters and initial-state artifact belong to the entire CVSim model; they are not offsets to apply to Shi/Hose.
- Source `eqns_ptr` sets the reported lower-body venous distending volume to zero for nonpositive transmural pressure, while `fixvolume_ptr` uses a negative `atan` value. The continuous reference rejects that branch. The observed source supine cohort has positive leg transmural pressure throughout.
- The superior vena cava Starling-resistor source leaves some branches unassigned: equality with the external pressure and reverse pressure when the outlet is above the external floor. The continuous reference admits only the three exact strict predicates that assign upstream `q[4]`, and rejects every other branch. The original observer additionally counts outlet/floor equality. Adding the complete admission check and rerunning all five trajectories preserved every raw and gzip trace byte.
- `fixvolume_ptr` computes a difference but its corrective pressure assignment is commented out. Original source drift is reported, never repaired by changing an initial pressure or target volume.
- Upper/lower-body and splanchnic regions are aggregate vascular beds. They do not establish separately calibrated brain, liver, muscle, skin, gut, gas-exchange or tissue-metabolism models. Supine source reference results do not qualify standing, walking, physiological calibration or native execution.

## Retained 2026-09-12 execution

The exact official source archive is [`cvsim-1.0.1.tar.gz`](https://physionet.org/files/cvsim/1.0.0/dist/cvsim-1.0.1.tar.gz), 17,540,984 bytes, SHA256 `8cd784628a1f5dbf7db36d00e6946d60c6546dd7f2fca67ae79242aacbfcdfd1`. The project version is 1.0.0; its archive filename is 1.0.1. All 17 retained C/header files match the corresponding archive members exactly.

Compact manifests, parameter and initial-state exports, compiler logs, reports and the independent audit are in [evidence/20260912](evidence/20260912). Complete compressed trajectories are in [the retained media directory](../../Docs/media/cvsim21-circulation-20260912). The 20-cycle regular traces have 34,287 rows and 70 columns: time; 21 P in mmHg; 21 V in mL; 24 Q in mL/s; total volume; atrial phase; ventricular phase.

The original unchanged 100-second source run accepted 100,382 steps. Its maximum departure from 5150 mL was 0.578230336 mL, and its cardiac phase differed from prescribed rational phase by up to 0.070816170 seconds. At identical original accepted times over 20 cycles, maximum original-versus-continuous differences were 12.4607902 mmHg, 7.36061047 mL and 628.047376 mL/s. These differences are retained; the references are not called numerically identical.

Tightening continuous tolerances from `1e-12` to `1e-13` changed pressure, volume and flow by at most `1.22498e-8` mmHg, `7.47565e-9` mL and `1.34563e-6` mL/s. Each group improved over the preceding `1e-10` to `1e-12` comparison. The tightest 20-cycle run changed total volume by at most `4.01633e-9` mL, and the minimum evaluated leg transmural pressure remained above 4.88302212 mmHg.

Native numerical evidence is checked by [analyze_cvsim_native.py](../analyze_cvsim_native.py), which independently reads the pinned C trace, checks every paired native/reference coordinate at exact cooked times, and retains all flow maxima in its timestep-refinement tests. The optional Heldt-table coordinate translation is explicit: add 184 mL to source volume index 2 and subtract it from index 5. The upstream reference remains unchanged.
