# Physical bed-load initialization diagnosis

SSH Numi, Apple M6. This is an offline static diagnostic for the retained 28,632-tetrahedron native shell, with 7,233 nodes in three layers. Input hashes in each receipt identify the geometry, mass, materials, native timestep and first-touch data. The floating seed uses vertical displacement and small pitch/roll. The skin/fat uses the published material parameters, native zero-velocity IPC normal contact and measured compiled mass. Deep nodes are prescribed relative to this seed for the diagnostic. The native scene uses 14 elastic attachment rows and requires its own coupled initialization and verification. No diagnostic forces or constraints were installed in the native simulation.

The measured total assembly mass is 97.131947 kg and its weight is 952.864403 N. First touch on the flat bed puts all first contacts on one side of the center of mass. The previous rigid tilt seed could place three support nodes but lost them in the native replay. Its retained 2 ms endpoint had just 0.401 N of support reaction. This diagnostic tested whether the shell can deform toward a better initial state.

| Case | Requested load | Reaction (N) | Active nodes | Result |
| --- | ---: | ---: | ---: | --- |
| Rigid bed | 5% | 47.6432 | 59 | Converged in the restricted static problem |
| Rigid bed | 6% | 48.5772 | 59 | Adipose cell 22244 reached the minimum volume ratio, 0.350001 |
| Flat mattress | 2.5% | 23.8216 | 205 | Adipose cell 20754 reached the expansion guard near 2.5, nodal residual 0.03343 N |
| Flat mattress with lateral freedom | 2.5% | 23.8216 | 207 | Same expansion witness, nodal residual 0.01784 N |

Matching the resultant is insufficient. The offline convergence gate also checks nodal force, root force and root moment. For the rigid bed at 5% load these residuals pass. For both mattress cases they fail, although the contact and passive foundation displacements agree within 0.4 nm. The flat mattress uses 10 kPa elastic modulus and Poisson ratio 0.30 as a reported modeling prior, plus an assumed 120 mm thickness. Its finite thickness and linearized normal response are modeling assumptions. [Published support model, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9284654/)

Cell 22244 has a reference determinant divided by cubed diameter of 0.079706. Cell 20754 has 0.000104904, and adjacent cell 20753 concentrates compression in the mattress runs. This points to two distinct defects under load, with the thin cell requiring loaded-state qualification of its local mesh neighborhood. The reference state and all reported loaded trials have positive cells. The maximum expansion or compression guard and force residual rejected the indicated trials before equilibrium. No native candidate is promoted.

Reproduce on Numi from the diagnostic directory beside the retained assets with NumPy and SciPy.

```sh
python3 test_predict.py
python3 predict.py rigid-partial-15 --loads .05 --max-iterations 145
python3 predict.py rigid-confirm-16 --loads .05 .06 .075 .1 --max-iterations 145
python3 predict.py mattress-confirm-17 --mattress-kpa 10 --loads .025 .05 .06 .075 .1 --max-iterations 145 --redirect
python3 predict.py mattress-lateral-confirm-18 --mattress-kpa 10 --lateral .05 --loads .025 .05 .06 .075 .1 --max-iterations 145 --redirect
python3 inspect_limits.py rigid-partial-15 rigid-confirm-16 mattress-confirm-17 mattress-lateral-confirm-18 > limits.json
```

Five tests exercise energy gradients and tangents, the admissible warm-start mapping and the mattress series contact. Exact deformed-cell ratios are independently recomputed in the saved runs. Candidate displacement arrays and failed attempts are in the separately verified private checkpoint. The surface-crossing work remains on the distinct 34,712-cell scalp candidate. Full gravity, shoulder behavior under load, motion clearance and the native 16-second variable-pleural run at 0.1× remain open.
