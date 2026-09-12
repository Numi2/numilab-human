# Rodero case18 material attribution, 12 September 2026

**PASS for source material attribution and exact LV geometric mass accounting.**
The [qualification receipt](media/cardiac-material-attribution-20260912/qualification.json)
binds the source, implementation, full runs and retained preparation failures.
The new [sidecar](../config/cardiac-material-rodero18.v1.json) assigns supported
passive material classes to every source tetrahedron and records unresolved
closure classes explicitly. It does not cook or simulate an anatomical wall.
The original [wall configuration](../config/cardiac-wall-rodero18.v1.json),
raw fields, geometry and historical receipts are unchanged.

The earlier blanket density-absence claim was too broad. Rodero section 3.1
estimates LV mass by multiplying summed LV mesh element volume by
**1.05 g/mL = 1050 kg/m³**, citing Vinnakota and Bassingthwaighte.
That supplies an LV geometric mass convention; it does not identify the CARP
inertial density or a subject-specific tissue measurement. All native inertial
densities remain null. No density is assigned to other labels or artificial
closures. [Rodero article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008851),
[density reference](https://journals.physiology.org/doi/full/10.1152/ajpheart.00478.2003).

## Explicit partial material map

The map joins the source label names in S1 to the passive tissue categories
and SI-converted parameters in S4. It is a declared interpretation of those
tables, not a recovered original CARP material-ID deck.
[S1](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1008851.s001&type=supplementary),
[S4](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1008851.s004&type=supplementary).

| Class | Source labels | Passive category | Cells |
| --- | --- | --- | ---: |
| 0 | 1, 2 | Ventricular Guccione | 1,097,534 |
| 1 | 3, 4, 18–24 | Atria and retained vein tissue | 249,344 |
| 2 | 5 | Aorta | 47,068 |
| 3 | 6 | Pulmonary artery | 17,914 |
| 4 | 7–10 | Artificial valve layers | 44,585 |
| `0xffffffff` | 11–17 | Unresolved inlet and removed-appendage closures | 13,638 |

Guccione parameters are `a=1700 Pa, bf=8, bt=3, bfs=4`.
The other classes use the source isochoric neo-Hookean form with
`c=7450, 26660, 3700, 1000000 Pa`, respectively. The source bulk penalty
parameter is `1e6 Pa`; its energy form differs between the two laws.
The closure labels have no explicitly attributed passive coefficient.
Their stiffness is not inferred from nearby vein or valve labels.
The artificial valve layers are not claimed to be anatomical leaflets.

Class IDs are **not** indices into `WorldSource::materials`. The
`HumanPack.cardiac-material-attribution.v1` format is deliberately partial;
`require_native_ready` rejects it even if a caller changes qualification flags.
A complete native assignment needs a separate admitted material/density owner.

## Deterministic implementation and execution

[`prepare_material_attribution`](../src/numilab_human/cardiac_material_attribution.py)
validates the exact source manifest, all thirteen source buffers and source
counts. It hashes consumed node, tetrahedron and label bytes, checks exact
positive cell determinants, and independently compares each regional count
and volume with the importer manifest. All source and implementation hashes
are checked again before publication.

The buffer contains 1,470,083 UInt32LE class IDs in unchanged source cell order.
The identity binds the parent and sidecar configuration, source manifest,
source buffers, output bytes and exact LV mass record. New output is published
atomically. Existing output is checked by streaming generated chunks against
the existing buffer, with no second staging buffer or output rewrite.
Changed inputs, manifests, class buffers and symlinks are rejected.
The [16 focused tests](../tests/test_cardiac_material_attribution.py), plus the
19 frame and 10 importer tests, pass locally and on the Mac mini.

The public command is:

```sh
numi human-cardiac-material-attribution --asset ASSET --output OUTPUT
```

The isolated full-source execution used:

```sh
PYTHONPATH=/Users/n/human-cardiac-material-attribution-20260912/source/src \
/Users/n/human-cardiac-partition-20260912/venv/bin/python \
  -m numilab_human.cardiac_material_attribution \
  --asset /Users/n/human-cardiac-wall-anatomy-20260912/asset-final \
  --output /Users/n/human-cardiac-material-attribution-20260912/attribution
```

[Full conversion](media/cardiac-material-attribution-20260912/source-attribution/full-source-attempt-001.execution.json)
passed in **4.0907 s**. The
[existing-output verification](media/cardiac-material-attribution-20260912/source-attribution/repeat-source-attempt-001.execution.json)
passed in **4.0811 s**, retaining both output hashes and modification times.
The captured maximum child RSS was **113,393,664 bytes**.
All inputs and implementation hashes matched before and after both runs.
These are offline source checks with **zero physical steps**.

## Exact LV accounting and independent comparison

For the **722,773 label-1 cells**, exact determinants of the unchanged
binary64 metre coordinates give **87.68573554497356 mL**. Multiplication by the
source convention gives **92.07002232222225 g**. The
[manifest](media/cardiac-material-attribution-20260912/source-attribution/attribution/manifest.json)
stores reduced exact rational volume and mass using hexadecimal integers.
This is a sum of source cell volumes: global tetrahedral embedding has not
been established, so it is not promoted to a proven geometric union volume
or blood/tissue mass partition.

The independent C++ reader checked all 1,470,083 assignments and recomputed
the exact LV volume and mass. Its
[full-source comparison](media/cardiac-material-attribution-20260912/cpp-evidence/full-source-attempt-001/)
passed in **0.8799 s**; all label/class counts and both exact fractions agree.
The C++ controls passed **12 valid and 9 invalid cases**. GMP's approximate
`get_d` result truncates where Python rounds to nearest, so the displayed
floating-point mass differs in its last bit; the exact fractions are identical.

The source's separate RV tissue volume discrepancy remains unresolved:
geometry yields **45.02490340997014 mL**, while the source CSV reports
**47.4406584127 mL**. No label remapping or rescaling is applied to conceal it.
The existing [geometry defect evidence](CARDIAC_WALL_ANATOMY_20260912.md)
also remains applicable.

## Retained identities and next gates

The [sidecar](../config/cardiac-material-rodero18.v1.json) pins the CC BY 4.0
[case18 source record](https://zenodo.org/records/4590294) and article evidence:

| Identity | SHA256 |
| --- | --- |
| Original wall configuration | `23c931fef53edced85e0e0a36c73d8490dddb87db6bd988482a2cdfc5a1442cc` |
| Imported asset manifest | `e8cb0391623577efc4eac04e5710cf7c9a4757614e09d936f5af3889c37d56f1` |
| New attribution configuration | `2091ac78031e5751f4062639b9408542765b5368e04d909420a2633d7242a8b1` |
| Main article XML | `9a458d885ad810d7ab309b37f4a5784112cf4c3e86e5dd0c6866a44c53eda252` |
| S1 PDF | `08777693dcfd5cfaac676c3e015761d62b68706af21577e678f89808a0c979ba` |
| S4 PDF | `5699feb01b7f91f93fd75e26a9a5043ca6bfb5da1803f010c45c71270992ebc6` |
| Class buffer, 5,880,332 bytes | `4c6aa8af284ea640e278c28d5a66dc8c579c2cc018bb71e41132d23871259080` |
| Attribution manifest | `3a7b0631d059295d4011a127ba1daaf75607cab5a6f24dd526fcc07381442571` |
| Attribution source identity | `f35cf2641869272d94891eec4533d44352b785571d3e111c93df1d7468e48515` |

The article XML is independently retrievable from
[Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8049237/fullTextXML);
the pin identifies the exact bytes used for the density correction.

Execution of anatomical mechanics still requires resolved closure materials
and inertial conventions, admitted wall/port geometry, an unloaded reference
or explicitly qualified loading construction, and quantitative support data.
The retained source describes pressure and activation settings, but supplies
no unloaded case18 mesh, complete Robin coefficient field or per-node
activation-time field. S4's five active-tension parameters do not establish
every parameter of an arbitrary current CARP implementation.
These missing inputs cannot be filled with software defaults and called
source reproduction. Source activation, physiological calibration, native
cooking and whole-Human completion remain unqualified.
