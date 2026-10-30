#!/bin/zsh
set -euo pipefail
H=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706
BASE=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209
S="$H/downstream-004-same-material-current-reference-229/native-domain-217-source-007"
P="$BASE/native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
C="$H/combined-child-handoff-142"
exec /Users/numi/numi-human-prep-venv-20261005/bin/python "$S/export_child_semantic_labels_217.py" \
  --parent-mass-csv "$P" \
  --parent-full86-weights-csv "$P.weights.csv" \
  --child-node "$C/skin-fat-combined.node" \
  --child-ele "$C/skin-fat-combined.ele" \
  --output-labels-csv "$H/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/child.labels.csv" \
  --source-map-csv "$H/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/child.labels.csv.source.csv" \
  --receipt-json "$H/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/receipt.json" \
  --capacity-profile full-reference-500k-nodes-2m-cells \
  --reference-registration-max-distance-m 0.02 \
  --reference-registration-ambiguity-tolerance-m 1e-8 \
  --p1-supported-reference-anchor \
  --p1-skin-support-npz "$H/current-semantic-atlas-088/skin-semantic-atlas-002/output-001/skin-current-semantic-support.npz" \
  --p1-skin-support-receipt-json "$H/current-semantic-atlas-088/skin-semantic-atlas-002/output-001/receipt.json" \
  --p1-fat-support-npz "$H/fat-current-semantic-support-141-root/fat-current-semantic-support.npz" \
  --p1-fat-support-receipt-json "$H/fat-current-semantic-support-141-root/receipt.json" \
  --combined-element-map-csv-gz "$C/skin-fat-combined-element-map.csv.gz" \
  --combined-node-map-csv-gz "$C/skin-fat-combined-node-map.csv.gz" \
  --combined-handoff-receipt-json "$C/adapter-receipt.json" \
  --anchor-csr-npz "$H/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/p1-anchor.npz" \
  --anchor-projection-csv "$H/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/p1-anchor-projection.csv" \
  --projection-helper-path "$H/batched_reference_tet_projection_149_root.py" \
  --projection-helper-sha256 02b04cff325b4adea24d055fdfd2121ab4d596a3c075ab6fe8183094eb46e1d6 \
  --collect-rejection-census \
  --rejection-witness-json "$H/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/rejection-witness.json"
