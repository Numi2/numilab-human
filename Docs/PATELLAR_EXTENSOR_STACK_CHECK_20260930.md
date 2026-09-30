# Patellar placement and extensor-stack regression check - 30 September 2026

The apparent posterior kneecap has two distinct source states. Literal MyoSim
`qpos0` puts the right and left patella body centers **8.086 and 8.088 mm
behind** their knee anchors. After the authored joint equalities are projected,
both centers sit **44.342 and 44.340 mm in front**. The source visual command
already selects the projected state by default; it now fails before rendering
if either projected center falls below a source-specific 25 mm anterior margin.
`--raw-source-rest` remains a diagnostic and records a failed display gate.

The separate decoded `NHBONES1` full-mesh audit remains stronger for the
compiled bone representation: every patella vertex is anterior in **16/16**
side/pose checks, with a smallest offset of **6.093 mm**. This is static
projected geometry evidence, not evidence from the historical 10-second
standing clip.

The Open Knee(s) compiler now also checks the gross extensor-layer order on
both left and mirrored-right payloads. Against the current source registration,
the patellar cartilage centroid is **8.739 mm posterior** to patellar bone;
quadriceps tendon is **42.142 mm proximal** to the bone; patellar tendon is
**46.179 mm distal**. The patellar bone centroid remains **46.934 mm anterior**
to the knee anchor and fibula **28.565 mm lateral** to tibia. Reversing any
of these three extensor layers fails compilation. Both 34 MB knee payloads
compiled on the CPU and passed the new gate. The [source-bound receipt](media/patellar-extensor-stack-20260930/receipt.json)
contains exact input, output and program hashes plus an independent
source-coordinate check.

The left source is subject `oks003`; the right remains its declared sagittal
mirror, not an independently segmented right knee. The expected chain follows
the [published description of the knee extensor mechanism](https://pubmed.ncbi.nlm.nih.gov/32119474/).

Reproduce the receipt with:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/verify_patellar_extensor_stack.py
```

This closes the **gross posterior-placement and reversed extensor-stack
presentation regression** for these source-bound static states. It does not
verify the facing normals of the patellar articular surface, patellofemoral
contact pressure, loaded tendon-force transfer, clinical subject anatomy, or
the old standing video's anatomy. Five source dependent-coordinate range
conflicts remain in the lower-limb pose audit. Broader placement is likewise
unfinished: 58 explicitly bilateral source surfaces pass the current
source-rest laterality gate, but the other 521 have no verdict from that gate;
seven of 28 tested pairs among eight abdominal organs have exact mesh
crossings. We do not move source organs or relax gates to hide those failures.
