# Direct matched-resting pair accepted-grid check (revision 002)

This bundle preserves the original direct-pair analyzer and an unlaunched sibling that closes a missing accepted-time-grid check. The original analyzer is pinned at `4b273d7c84ab3aba0ebc1a55ab912c4e71c02cfb650493ab7813c7e08a7b31d1`; revision 002 is pinned at `3856e0a0c2de8a95d5f9ebf62354d827946b4a375fc348b777b015ac6d423523`. Revision 002 checks the expected 8-step accepted CSV grid, integer/increasing steps, each timestamp against the declared Float32 `dt=0.002` serialization, and exact equality of the parsed `(step,time_s)` sequence between the two arms before any window summaries run. The paired declaration remains a measurement plan; this package does not execute it.

The packaged nine-test regression command passed on Mac mini with Python 3.9.6:

```sh
cd /Users/n/numi-human-free-apex-two-family-1178
/usr/bin/python3 -m py_compile tools/evidence/native-resting-pair-grid-1230/source/analyze_direct_pair_revision-002.py tools/evidence/native-resting-pair-grid-1230/tests/test_analyze_direct_pair_grid_revision_002.py tools/evidence/native-resting-pair-grid-1230/runners/validate_current_control_grid.py
/usr/bin/python3 tools/evidence/native-resting-pair-grid-1230/tests/test_analyze_direct_pair_grid_revision_002.py
```

The current closed 1218 control trace was independently checked with:

```sh
cd /Users/n/numi-human-free-apex-two-family-1178
/usr/bin/python3 tools/evidence/native-resting-pair-grid-1230/runners/validate_current_control_grid.py --output tools/evidence/native-resting-pair-grid-1230/reports/current-control-grid-check-revision-002.json
```

The retained runner refuses to overwrite an existing report; reruns require a fresh output path. `current-control-grid-check-attempt001.json` is retained as the first report; its calculations were unchanged, but its nested comparison text did not clearly label the single-trace self-check. Revision 002 corrects that scope wording. It has 19,375 expected rows, steps 8 through 155,000 at stride 8, time range 0.01600000076 to 310.000014724 s, and sequence digest `451f5d9d6bcf8edb563708e55950d037c83f27ae87e58331e37216cbaa52162b`. Its CSV SHA-256 is `e06988dbcf8a187bab35b854d80d1a0e5111a4b7f90a286af9d9614fa5d952d5`. This is a single-control grid check, not a paired-grid comparison or treatment result.

The direct analyzer's planned treatment is a 0.5 drive scale over `[60,100)` s. It summarizes pre `[30,60)`, dose `[70,100)` (the final 30 s of the drive window), and recovery `[280,310)` s. Its declared source, invocation, runtime, and asset checks are retained in the source copy. The direct analyzer computes measured differences but does not apply response/recovery pass thresholds. No new thresholds are introduced here.

No current treatment declaration or treatment run exists. The current 1218 baseline scene is not anatomy-cleared: the retained 95 s audit reports 18 skin-target crossings, and the four late captures report 845, 890, 907, and 853 pairs. Therefore this bundle does not authorize or imply a current-scene paired outcome. The owner-CLI command-only attempt is retained with its exact failure: exit 1 because the owner CLI command differed from the baseline outside output/capture/intervention settings.

The completed 1170 registered pair remains prior evidence, not a substitute for a current-scene pair. Its registered analysis SHA is `1936c4d4c54120ef230b67cf9cdbcb06656233e01c6637bc3e8fa57c297e1018`; its paired report SHA is `745a7f0698d729919851923dd6dbbee264724c2130792b38b68ab6e9dc142ec7`. That report measured a PaCO2 difference-in-differences of +1.835245 mmHg and dose-window inspiratory ventilation treatment-minus-control of −1.3424 L/min; its recovery differences were within the then-registered numerical margins. It was one deterministic registered pair, not population inference or clinical validation. It used older 1159-era skin, soft-tissue, thorax, and respiration assets than the current 1218 baseline, and it lacks the direct analyzer's per-run `run-metadata.json` interface. The current trace happens to have the same CSV hash as the old registered control trace, but that does not make the old treatment a matched outcome for the current scene.
