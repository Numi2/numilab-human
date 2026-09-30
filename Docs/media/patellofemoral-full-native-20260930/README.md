# Full-source patellofemoral native preflight

This internal receipt binds the pinned Open Knee(s) PTC/FMC mesh to a bounded
Numi Matter step. It is a development diagnostic, not an adopted knee pose or
loaded-knee qualification. The original bilateral cartilage configuration has
18 exact full-boundary face crossings per side and fails the native initial
crossing gate with bitwise rollback. The unadopted 20 µm PTC translation removes
those face crossings and completes one 1 µs, zero-gravity step with a prescribed
0.1 m/s approach velocity and 20 µm contact slop. Contact-on and contact-off
accepted states are retained for both sides in the four gzip files.

To verify the retained states against the pinned source mesh on a machine with
the Human Python dependencies:

```sh
PYTHONPATH=src:. .venv-mujoco312/bin/python tools/audit_patellofemoral_full_native_step.py
```

To recapture the native experiment on Apple silicon, build the
`numi-matter-patellofemoral-full-surface-step` target in Numi Lab and run:

```sh
PYTHONPATH=src:. .venv-mujoco312/bin/python tools/audit_patellofemoral_full_native_step.py \
  --capture --matter-root /path/to/NumiLab --build-dir /path/to/NumiLab/build
```

The capture command regenerates the source-bound cartilage inputs, verifies a
14 µm ambiguity rollback on both sides, runs the bilateral 20 µm contact-on/off
comparison, and checks a byte-identical left replay. The final native surface
guard is intentionally conservative where FP32 cannot distinguish a very close
vertex from a short penetration. Exact integer face tests on every retained
accepted state found zero PTC/FMC crossing or point-contact face pairs.

The FMC source boundary has a two-fan vertex link at source node 233523, so
these face tests do not establish disjoint solid volumes. The step excludes
patellar bone and PTB/QAT/PTL attachment, uses synthetic density and an
unvalidated isotropic cartilage energy, and has no physiological load or
whole-system energy closure. Eleven active contact histories are not a force or
pressure measurement; the accepted barrier impulse field was zero. No clinical
or loaded-knee claim follows from this receipt.
