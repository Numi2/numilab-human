# Shi–Hose source reference

This is an independent FP64 reference for the complete four-chamber, systemic and pulmonary **hydraulic** circulation in the curated Shi–Hose CellML model. It is an offline evidence tool. Matter remains the live simulation owner.

The source is *Zero dimensional (lumped parameter) modelling of native human cardiovascular dynamics*, by Yubing Shi, Rod Hose and Physiome Model Repository contributors, revision `a679cdc2e97429fb5280af8132c119758626c1f2`. The [curated exposure](https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5) grants [CC BY 3.0](https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5/ModelMain.cellml/license_citation). Numi's changes translate the source representation and add an independent numerical integrator and evidence export. The original CellML authorship metadata is retained in `third_party/physiome/shi_hose_2009`.

The [immutable archive](https://models.physiomeproject.org/workspace/shi_hose_2009/@@archive/a679cdc2e97429fb5280af8132c119758626c1f2/tgz) has SHA256 `91d6b586c0caaa0fd59ec21cef873f1348563af0033210fbadfe84921a52782d`. `source-lock.json` records every imported CellML file digest, the archive identity and the saved license-page identity. The generator rejects changed or unpinned imports.

## Reproduce

Run from the Human repository. Set `NUMI_MATTER_SOURCE` to the matching native source checkout containing `matter/tools/shi_hose_reference/`.

```sh
python3 tools/shi_hose_reference/generate_reference.py --source-dir third_party/physiome/shi_hose_2009 --output-dir Build/shi_hose_reference
python3 tools/shi_hose_reference/generate_observables.py --output-dir Build/shi_hose_reference
python3 tools/shi_hose_reference/verify_generation.py --source-dir third_party/physiome/shi_hose_2009 --native-header-dir "$NUMI_MATTER_SOURCE/matter/tools/shi_hose_reference"
cp "$NUMI_MATTER_SOURCE/matter/tools/shi_hose_reference/shi_hose_dopri.hpp" Build/shi_hose_reference/
c++ -std=c++23 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror -I Build/shi_hose_reference tools/shi_hose_reference/reference.cpp -o Build/shi_hose_reference/reference
Build/shi_hose_reference/reference Build/shi_hose_reference/source-20cycles-1e10.csv 20 .001 1e-10 > Build/shi_hose_reference/source-20cycles-1e10.json
Build/shi_hose_reference/reference Build/shi_hose_reference/source-20cycles-1e12.csv 20 .001 1e-12 > Build/shi_hose_reference/source-20cycles-1e12.json
Build/shi_hose_reference/reference Build/shi_hose_reference/source-20cycles-1e13.csv 20 .001 1e-13 > Build/shi_hose_reference/source-20cycles-1e13.json
python3 tools/shi_hose_reference/compare_traces.py Build/shi_hose_reference/source-20cycles-1e10.csv Build/shi_hose_reference/source-20cycles-1e12.csv Build/shi_hose_reference/source-20cycles-1e13.csv Build/shi_hose_reference/reference-refinement.json
```

`generate_reference.py` follows actual imports, encapsulation, connection aliases, parameter-valued initial states and MathML equations. It produces 14 ODE states, 137 alias-unified variables and 34 algebraic equations from 27 component instances in 15 CellML files. It does not call the Human hydraulic compiler or Matter equations. Python only translates source and compares existing outputs; all physical stepping uses C++ Dormand–Prince 5(4), with the coefficients of [Dormand and Prince (1980)](https://doi.org/10.1016/0771-050X(80)90013-3).

The source coordinate units are retained during integration. Observable export uses the exact source `UnitP = 133 Pa` and `UnitV = 1e-6 m3`; the activation equations retain the literal `3.14159`. Initial ventricular volume is the source `V0 = 500 mL`, not the pressure–volume reference `Vini`. There is no smoothing, state clipping, modified initial pressure or invented vascular volume offset.

## Mapping and evidence

`native-state-mapping.json` maps the 36 named SI observables and 20 native hydraulic coordinates to the generated source variable indices. The native coordinates are, in order:

- Storage: LA volume, LV volume, SAS compliance storage, SAT compliance storage, SVN compliance storage, RA volume, RV volume, PAS compliance storage, PAT compliance storage, PVN compliance storage.
- Flow: mitral, aortic, SAS outlet, SAT outlet, SVN outlet, tricuspid, pulmonary valve, PAS outlet, PAT outlet, PVN outlet.

Storage is in m3 and flow in m3/s. The six vascular coordinates are `C*P`; their initial values are determined by the source pressure and compliance. They are not absolute whole-blood volumes. The four chamber coordinates are actual source chamber volumes. Series zero-storage arteriole/capillary resistances remain present in the generated source equations and observables.

The recorded 2026-09-12 run covers the entire initial transient and 20 one-second cycles at a 1 ms output interval. `evidence/20260912` contains exact build, source and artifact identities, step counts, pairwise refinement statistics, and initial/final coordinates. Between relative/absolute source-coordinate tolerances `1e-12` and `1e-13`, the maximum differences were 9.83e-6 Pa in pressure, 2.84e-13 m3 in chamber volume and 5.03e-11 m3/s in flow. These are measured pairwise differences, not rigorous exact-solution error bounds. All three families improved from the coarser comparison.

At tolerance `1e-13`, total source hydraulic storage drift was below 1.94e-17 m3. The last-cycle state change was 2.18e-5 in `abs(new-old)/(1+abs(new))` source coordinates, so the 20-second endpoint is not declared perfectly periodic. The trace retains the source's initially negative venous return and large initial ventricular volumes.

This establishes a reproducible numerical reference for this curated CellML revision. It does not reproduce the original papers' dynamic valve-leaflet model: this revision uses instantaneous one-way square-root valves. It does not establish absolute vascular blood volumes, species dilution volumes, gas exchange, metabolism, tissue perfusion calibration, patient calibration, organ mechanics or whole-human biological completion. Native agreement is a separate qualification and must report its own timestep, solver and acceptance evidence.
