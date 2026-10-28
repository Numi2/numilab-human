# Registered skin/fat volume node-map preparation

This isolated staging bundle contains the combined TetGen node-map producer and the existing child-cell semantic-label exporter it loads. Keep all four Python files in the same directory: register_combined_volume_nodes_217.py locates export_child_semantic_labels_217.py beside itself. This version has no --label-owner-dir option.

The producer extends the existing combined node-map CSV with typed O/I/D and interior-node registration columns. Boundary identities and donor-frame ancestry come from the existing registered-surface and parent-registration tables. Interior rows use a bounded current-parent tetrahedron registration restricted to the incident child cell's material, geometric donor, and anatomical region. The exporter writes the existing five-column child semantic labels from parent provenance.

## Staged checks

Run from this directory on the Numi host:

    STAGE=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/registered-volume-node-map-232/human-tools-staging
    PY=/Users/numi/numi-human-prep-venv-20261005/bin/python
    W=/Users/numi/Work/skin-fat-native-integration-209

    cd "$STAGE"
    "$PY" -m py_compile export_child_semantic_labels_217.py test_child_semantic_labels_217.py register_combined_volume_nodes_217.py test_register_combined_volume_nodes_217.py
    "$PY" -m unittest -v test_child_semantic_labels_217.py
    "$PY" test_register_combined_volume_nodes_217.py "$STAGE/producer-test-output"
    "$W/build-native/numi-human-physical-skin-registered-volume-csv-check" "$STAGE/producer-test-output/fixture-registered-node-map.csv" 7 157
    "$W/build-native/numi-human-physical-skin-registered-volume-import-check"

The retained run outputs are in verification/. The exporter regression is an integration check requiring the native parent capture and its four companion CSV files. Set `NUMI_HUMAN_PARENT_MASS_CSV` to another verified capture to run it elsewhere. Its retained default is:

/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv

## Producer invocation

The retained producer fixture can also exercise the staged command-line entry point:

    OUT="$STAGE/producer-test-output"
    CLI_OUT="$STAGE/cli-test-output"
    "$PY" register_combined_volume_nodes_217.py \
      --node-map "$OUT/inputs/skin-fat-combined-node-map.csv.gz" \
      --node "$OUT/inputs/combined.node" \
      --ele "$OUT/inputs/combined.ele" \
      --labels "$OUT/inputs/child.labels.csv" \
      --parent-base "$OUT/inputs/compiled-parent-cell-mass.csv" \
      --registered-dir "$OUT/inputs/registered" \
      --poses "$OUT/inputs/poses.json" \
      --maximum-parent-distance-m 0.02 \
      --ambiguity-tolerance-m 1e-10 \
      --output "$CLI_OUT/fixture-registered-node-map.csv" \
      --receipt "$CLI_OUT/fixture-receipt.json"

## Scope

The fixture is a small adapter regression: seven nodes, four positive Float32 tetrahedra, three shared I interface rows, and two interior nodes. It covers shuffled source IDs/map rows, direct and face source identities, a far alternate semantic parent with a valid near incident alternative, and wrong O/D partition and parent-tuple rejection.

The receipts set physical_admission to false. These tools do not admit a production child volume, establish geometric ancestry for interior nodes, transfer mass, or qualify physical motion. A changed-volume candidate still needs its own valid positive mesh, complete semantic labels, registered boundary/facet maps, strict native admission, and physical verification.
