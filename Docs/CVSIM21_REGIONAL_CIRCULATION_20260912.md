# CVSim21 regional circulation — source and ownership record

This increment authors a separate CVSim21 aggregate circulation with sourced absolute blood-volume coordinates. It contains 21 pressure/volume compartments, 24 algebraic flows, four cardiac chambers, four systemic regional beds and a pulmonary bed. Its source model has 5150 mL total blood and 4166 mL total zero-pressure filling volume. These are model parameters, not measurements of the Human anatomy or a particular person. This graph does not supply offsets to the Shi/Hose model.

## Authoring and source identity

`numi human-circulation` dispatches to [cvsim21.py](../src/numilab_human/cvsim21.py). The default [configuration](../config/cvsim21-source.v1.json) selects `upstream_equation` volume coordinates; the separately selected [table-aligned configuration](../config/cvsim21-heldt-table-aligned.v1.json) selects `heldt_table_aligned`.

```sh
numi human-circulation --output Build/cvsim21/upstream.native.json
numi human-circulation --config config/cvsim21-heldt-table-aligned.v1.json --output Build/cvsim21/table-aligned.native.json
PYTHONPATH=src python3 -m unittest discover -s tests -p test_cvsim_parameters.py -v
```

The output is `HumanPack.physiology-native.v3`, qualified only as a `source_model_variant` authoring target. Python verifies and lowers data; Matter owns physical stepping. Graph identities bind the exact source manifest, authored configuration and compiled payload. Configuration explicitly fixes arterial baroreflex, cardiopulmonary reflex and tilt off. Numerical scales and tolerances carry their own numerical-control role and units.

The [source lock](../third_party/physionet/cvsim21/source-lock.json) pins 17 original C/header files and their enclosing PhysioNet archive. Actual initialization calls `initial_ptr`, `mapping_ptr` and `estimate_ptr`; the different legacy `initialization.c` supplies no defaults here. The restricted [parameter parser](../src/numilab_human/cvsim_parameters.py) reads all 153 mapped values and 568 scalar entries in 142 source-table quadruples. It retains nominal, dispersion and range metadata, multiplication provenance, unused members and exact source lines. It rejects duplicate, missing, nonfinite, unresolved and unsupported assignments without executing C. Its 13 tests passed, including an independent compiled-C comparison of every mapped value and every table scalar.

Source ownership, equations, incidence and discrepancies are retained in [source-mapping.json](../tools/cvsim21_reference/evidence/20260912/source-mapping.json) and [the parameter metadata](../tools/cvsim21_reference/evidence/20260912/cvsim-source-parameters.json). The immutable [original initialization](../tools/cvsim21_reference/evidence/20260912/original-initial.json) preserves the original C pressure, volume, flow and parameter vectors. All initial storage and flow laws are independently checked during authoring.

## Equations and explicit model differences

Source units are mL, mmHg and seconds. Lowering uses `1 mL = 1e-6 m³` and the declared rounded conversion `1 mmHg = 133.3224 Pa` from [NIST Handbook 44, Appendix C](https://nvlpubs.nist.gov/nistpubs/hb/2018/NIST.HB.44-2018.pdf); it does not reuse Shi/Hose's 133 Pa source unit.

For a linear vascular compartment, `V = V0 + C(P − Pexternal)`. The splanchnic, leg and abdominal veins use a saturating displacement with `k = πC/(2Vmax)` and `V − V0 = (2Vmax/π) atan(k(P − Pexternal))`. Native inversion requires positive absolute volume and the open domain `|V − V0| < Vmax`; a positive `V0` alone does not establish positivity at arbitrary pressures. Four cardiac chambers retain the source compliance extrema and cosine activation shape.

There are 18 signed resistors, five forward linear valves and one Starling resistor. CVSim's linear valves are distinct from the square-root orifice law used by Shi/Hose. The incidence audit independently extracted every flow term from the 21 pressure derivatives: each of the 24 edges enters one compartment and leaves one compartment, with connected rank 20. This structural cancellation does not by itself qualify a numerical trajectory.

Three differences from the original executable are explicit for both volume-coordinate options:

- The native target uses a continuous fixed-rate clock with exact nominal period `6/7 s`. It replaces the original discrete SA-node scheduling algorithm; the original and continuous CPU references remain separate.
- Original `eqns_ptr` reports zero leg distending volume at nonpositive transmural pressure while its pressure derivative uses positive atan compliance. The native variant uses the continuous signed atan storage law. Original output evidence remains unchanged.
- Original Starling flow has unassigned equality and reverse-pressure branches that can retain the previous flow. The native stateless extension uses `Q = max(Pup − max(Pdown, Pexternal), 0)/R`.

The original `fixvolume_ptr` computes a discrepancy but its pressure correction is commented out. No correction is enabled by authoring, and no initial pressure is adjusted to conceal drift. Six source-table nominal values lie outside their declared ranges and remain flagged. Unused aggregate parameter 76 is 715 mL, whereas the eight arterial filling offsets sum to 772 mL; parameter 76 is not another blood-volume owner.

## Two explicit volume coordinates

The original mapper crosses the upper-body and descending-aorta filling offsets. Its equation output therefore assigns 16 mL to the upper-body arterial baseline and 200 mL to the descending-aorta baseline. The source hemodynamic table labels and [Heldt's 2004 thesis, Table A.1, page 157](https://puerta.csail.mit.edu/pdf/HeldtThesis04.pdf) support the opposite ownership: descending aorta 16 mL, upper-body arteries 200 mL.

| Coordinate selection | Upper-body arteries, source row 2 | Descending aorta, source row 5 |
| --- | ---: | ---: |
| `upstream_equation` — default | 16 mL baseline | 200 mL baseline |
| `heldt_table_aligned` — explicit composition | 200 mL baseline; volume shift +184 mL | 16 mL baseline; volume shift −184 mL |

The paired translation changes both `V` and `V0`, preserving pressure, flow and summed volume in this hydraulic model. Individual absolute volumes, dilution concentrations for fixed amounts and residence times change. The table-aligned option is not unmodified C volume reproduction or a calibrated anatomical partition. The mapping records the verified thesis page and PDF digest; the local public-PDF acquisition required an expired-certificate exception, which remains disclosed in its provenance. The thesis binary is a local research artifact, not vendored source code.

## Physical ownership and attribution

Compartment identifiers are explicit CVSim source aggregates. They have no invented FMA or BodyParts3D registration. Each source volume has one graph owner; those tokens do not prove that arbitrary anatomical regions are physically disjoint. The [18-region anatomical template](../config/physiology-organ-network-template.v1.json) remains a separate unresolved authoring input.

This hydraulic graph adds no FEM or rigid mass and supplies no blood density, mesh integration region or mechanical inertia partition. It contains no species amounts, oxygen chemistry, tissue reservoirs, exchange clearances, consumption, organ-specific calibration or reflex/tilt coupling. Regional renal or splanchnic flow labels do not establish a separately calibrated kidney, liver, gut or tissue-perfusion model.

Attribution belongs to the original CVSim authors and PhysioNet, as retained in the [source notice](../third_party/physionet/cvsim21/NOTICE.txt) and [reference documentation](../tools/cvsim21_reference/README.md). PhysioNet designates ODC-By-1.0 for the project. The source lock separately records that a model-level software grant has not been independently established; it does not relabel the C files as MIT, GPL or Apache. The Heldt thesis remains separately attributed.

## Native cohort evidence

The physical M4 Pro passed all five native cohorts with two bitwise-identical environments, exact accepted rational cardiac time, mid-trajectory snapshot replay and zero failed steps. Every source, executable, input and external Metal library hash matched before and after each final run. Native publication is `b91fe6832813497ab532e5dbe05ed8c8233e6b1f` on `origin/coupled`, with ABI28/package13 and unchanged snapshot/proof shape6.

| Volume coordinates / run | Steps | Timestep | Maximum volume error | Maximum flow error | Scaled state RMS error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream / one cycle | 429 | 2 ms | 0.460673 mL | 53.0559 mL/s | 0.0125879 |
| Upstream / one cycle | 858 | 1 ms | 0.235197 mL | 27.0312 mL/s | 0.00632401 |
| Upstream / one cycle | 1716 | 0.5 ms | 0.118181 mL | 13.9093 mL/s | 0.00320187 |
| Upstream / ten cycles | 4286 | 2 ms | 0.460673 mL | 53.0559 mL/s | 0.0118954 |
| Table-aligned / one cycle | 858 | 1 ms | 0.235054 mL | 27.0394 mL/s | 0.00632463 |

Times use the actual binary32 timestep: all short refinement runs end at 0.8580000407528132 s, slightly longer than one nominal 6/7-second cycle; the long run ends at 8.572000407148153 s, slightly longer than ten cycles. The independently recomputed volume/flow/RMS error ratios are 1.94–1.99 for each halving, passing the fixed numerical refinement gate of 1.5. The long run's maximum relative total-volume drift is 8.20e-7 (about 0.00423 mL out of 5,150 mL). These are numerical regression results, not experimental accuracy or physiological calibration bounds.

The independent continuous FP64 reference runs twenty nominal cycles. Tightening tolerance from 1e-12 to 1e-13 changes maximum pressure, volume and flow by only 1.23e-8 mmHg, 7.48e-9 mL and 1.35e-6 mL/s. Its volume sum changes by about 4.02e-9 mL. By contrast, the unchanged original C executable runs 100 seconds with maximum volume discrepancy 0.578230 mL and cardiac phase discrepancy up to 0.0708162 s relative to continuous rational time. All complete original and continuous trajectories are retained; source-clock equivalence is explicitly false.

The Human source, analyzer and receipt suite passes 59 tests; the existing 28 physiology, 17 cardiac-authoring and 23 historical-receipt tests also pass. All 15 native tests pass: strict v1/v2/v3 input admission, the new regional constitutive laws, legacy cardiac laws, vascular transport, stateful FEM/MPM, multiphysics, rollback, snapshot archives and accepted-state proof. A fresh full four-run Shi–Hose qualification also passes its unchanged numerical ceilings and refinement gates on ABI28. Its historical receipt and verifier remain byte-identical. The initial compiler failure at an atan-domain endpoint and a probe compilation failure remain recorded alongside the fixes and final passing checks.

The [receipt](media/cvsim21-circulation-20260912/receipt.json), [independent numerical report](media/cvsim21-circulation-20260912/native-comparison.json) and [verification tool](../tools/verify_cvsim21_20260912.py) bind these results to the authored graph, exact source, source execution, native publication and all retained trajectories. The suite command on the Mac mini and the direct local workspace command produce the exact same payload. At qualification, the local global `numi` symlink pointed to an iCloud dataless file and timed out; that observation remains in [the dispatcher record](media/cvsim21-circulation-20260912/suite-dispatcher.json). After publication, requesting download of the existing launcher through Foundation restored local dispatch without source changes. Running `numi human-circulation --output Build/absolute-blood-20260912/local-dispatch-recovered.native.json` then succeeded with the same payload SHA-256, `eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818`. The frozen qualification receipt and its original environment observation remain unchanged.

## Gates after the bounded hydraulic increment

1. Establish an explicit source-to-anatomy registration and disjoint blood-volume partition. Decide how blood and tissue volumes relate to existing FEM/rigid mass before introducing density or inertia; preserve each volume and mass owner exactly once.
2. Reproduce and qualify source-level reflex and tilt behavior with their own clocks, histories, boundary pressures and volume-loss transactions. Supine fixed-rate evidence does not transfer to standing or walking.
3. Source organ-specific vascular volumes, arterial/venous connections and perfusion targets. Aggregate regional beds require explicit subdivision and conservation accounting before individual-organ claims.
4. Source species amounts, accessible tissue volumes, exchange laws, partition coefficients, metabolic sinks and their uncertainty. Extend blood/tissue conservation and accepted-state rollback together, then assess physiological calibration against independent data.

These gates remain separate from whole-Human, biological and production qualification.
