# Native thickness-gated bed-layer repair

Apple M6 on SSH Numi. Source and engine pins are in the build and launch
receipts. The same compiled executable runs the unchanged control and the
opt-in two-node interior edit. Material, topology, contact, gravity,
ligaments and the pleural-pressure owner use the same physical code.

The first native attempt correctly rejected the original offline fit. One
adipose column measured 10.0030 mm, above its original 10 mm maximum.
The authoring projection now reads the original bounds and leaves 0.2 µm
for emission rounding. The repaired column measures 9.9998 mm. A second
attempt reached native compilation and the mass ledger before its old
visual pack failed authentication. Regenerating the pack from the
candidate geometry made that replay honest. That exact pack is pinned
and used by the extended run.

The emitted reference and current coordinates change only rows 3703 and
4034. The exterior and deep layers, 28,632 tetrahedra and typed anatomical
ancestry are identical to the control. Minimum shape is unchanged in the
two startup frames and all volumes are positive. Native skin/fat mass
goes from 13.939679828294647 kg to 13.939679828346488 kg. The largest
compiled mass difference across 628 material/region/donor banks is
1.8741e-10 kg. A 12-face exact Float32 audit detects zero new shell pairs
and zero bone pairs involving those changed interface faces.

Both four-step native runs completed. The repaired shell then
completed 128 consecutive steps covering 2.000 ms. Each audited
mechanical and vascular residual passed its active solver threshold.
All 34 registered ligament rows pass the terminal native audit. Every
shell cell has positive terminal volume, with minimum J 0.92720. An
exact audit of all 14,511 shell boundary/interface faces finds zero
self- or cross-layer pairs at the initial and accepted terminal states.
Bone crossings elsewhere and between captures remain outside this
result. The pleural trace varies between -490.332581 and -490.332489 Pa,
which exercises propagation with too little time for a full breath.

Physical support remains insufficient. Terminal bed reaction is
0.40419 N on three nodes versus 952.864 N total weight. No root assistance
was used. The body path took 99.073 s at 0.00002019×. Launch to exit took
113.723 s at 0.00001759×. These are headless startup measurements. The
16-second, zero-crossing, variable-pressure native qualification at
0.1× and the full 1× goal remain open.

The 14 finite attachments total 924.29 N/m stiffness. Even if they all
acted vertically, they alone would need at least one extension of
0.883 m to support the 816.12 N payload outside the shell. An extension
of one local fat thickness at each sample gives an upper bound of
8.47 N. The terminal sum over all actual tissue-to-body contact forces
was 0.561 N during the repaired replay. The existing triangle law starts
producing force only after signed penetration. A supported candidate must
show substantial load through real, distributed anatomical interfaces at
the required geometric clearance. The short release and spring bound do
not measure a settled state's contact capacity.

Six tests pass, including the regression for the rejected thickness
proposal. The complete native logs and compiled audit ledgers are in the
independently verified private archive. The exact pack can also be
restored from the local verified zstd copy or its private release asset.

```sh
python3 ../fit_preload_cells.py ../fit-thickness-23
python3 build_trial.py --fit ../fit-thickness-23/candidate.npz --compile
python3 run_trial.py candidate-fresh --candidate
python3 run_trial.py control-fresh-build
python3 run_trial.py candidate-128 --steps 128 --candidate --reuse-candidate-pack-from candidate-fresh
python3 verify_trial.py candidate-fresh verification.json
python3 verify_motion.py candidate-128 motion-verification.json
```
