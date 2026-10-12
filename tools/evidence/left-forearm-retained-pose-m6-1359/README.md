# Left forearm across 25 retained poses

Result: **passed**. The source-preserving diagonal repair 097 was deformed through all 25 retained body-pose snapshots on SSH numi. Each pose checks both left forearm muscles and their joined external surface: 75 complete exact Float32-lattice checks for self-intersections, degenerate faces, closed topology and component volume signs.

These are CPU reconstructions using the existing ordinary FMA deformation path and saved native poses. They are **not 25 new GPU captures**, do not qualify external anatomical interfaces, and do not complete the five-minute integrated deliverable. The CPU predictor is not bit-identical to the GPU. Actual 16-second native 012 evidence remains in [1358](../native-anatomy-left-forearm-diagonal-m6-1358/README.md).

The grid includes eight parent-run poses, one posture-sensitivity pose, and eight poses each from the retained baseline and respiratory-intervention runs. Late poses extend to accepted step 155000 from the older 310-second runs. Those older runs supply kinematics only; they are not final native evidence for the repaired anatomy.

Source geometry, field coefficients and physical state were not changed by this audit. Both muscle identities, laterality and raw-source provenance remain those documented in 1358. The two new diagonal faces have pooled derived-face ancestry documented there. This is a mixed-source reference adult, not a measured individual.

Reproduce on the retained numi host:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/audit_left097_retained25_cpu_001.py
```

The script refuses to overwrite an existing result directory. Preserve this executed script and choose a new output directory in a successor copy for reruns. All source/helper/pose inputs and copied outputs are SHA-256 pinned in publication.json. Component volume signs are checked separately in component-volume-sign-audit.json; they are not a physical mass or density validation.

Right-forearm repair, back seams, quadratus/neighbor interfaces, and final integrated baseline/intervention recording remain open. No runtime code or performance claim is changed by this evidence increment.
