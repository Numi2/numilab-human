# Replay conserved skin and fat mass against the final repaired mesh

The production source readers and feasibility replay accepted the exact reference mesh from the registered-volume preflight. The input includes 311,020 nodes, 1,448,804 cells and 28,632 immutable source tets. Only three nodes needed new P1 coordinate proofs and their parent maps matched exactly. Thirty-four centroids changed and 26 carrier witnesses were updated. All six compensation parents and 6,184 compensation cells were retained unchanged.

The independent native replay verified 13.944580336345139 kg, 5,336,421 expanded routes, a maximum source-cell error of 1.31e-15 kg, and a maximum implied transport distance of 0.18744398407466167 m under the existing 0.187444 m limit. Both material density ranges passed with zero violations. The production writer and reader round-trip passed for both role flow sets. No physical steps were executed in this audit.

Run the retained preparation tools on SSH `numi` from the paired reference working directory. The low-level source and topology pins are retained in the preregistered checkpoints.

```bash
"$HOME/numi-human-prep-venv-20261005/bin/python3" revalidate_local_p1_for_volume_floor.py
"$HOME/numi-human-prep-venv-20261005/bin/python3" rebind_floor_carrier_witnesses.py
"$HOME/numi-human-prep-venv-20261005/bin/python3" close_native_density_roundoff.py --reference-floor
mkdir paired-floor-native-role-flow
NUMI_HUMAN_PAIRED_REFERENCE_FLOOR=1 \
NUMI_HUMAN_NATIVE_ROLE_FLOW_OUTPUT="$PWD/paired-floor-native-role-flow" \
  /bin/bash run_complete_fine_p1_native_check.sh --full-replay
```
