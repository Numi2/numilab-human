# Quality repair evidence: 740 and 752

This package preserves the full-neighbor 740 quality attempt, its independent 741/748 replay, and the independent 752 fixed-boundary journal replay promoted by 754. It is an evidence snapshot. The source files and reports are copied byte-for-byte; volume arrays and source-state inputs remain external at the exact paths and hashes recorded in `manifest.json`. The scripts retain their original workspace-relative assumptions and are not a portable standalone runner.

## Recorded results

- 740: the full-neighbor batch reduced native quality failures from 83 to 82 for Skin and 477 to 468 for Fat. Independent replay 748 verified the fixed-boundary candidates at 82 Skin and 468 Fat. The batch status is `PASS_WITH_PARTIAL_NATIVE_DEFECTS`.
- 752: the independent fixed-boundary replay verified the point stage and journal. Skin had zero flip operations and remained at 81 bad cells. Fat used six operations and improved from 468 to 467 bad cells.
- Every included receipt records `physical_admission: false` and `native_steps: 0`. These results do not establish native solver, mass-allocation, or clinical qualification.

## Reproducibility boundary

The manifest records the exact source-workspace paths, content hashes, and byte sizes for external volume arrays and parent states. Replays require those pinned source inputs and the original Numi host layout. No candidate NPZ arrays are silently substituted or claimed to be included. The 747 Steiner proposal is intentionally outside this package; it is not promoted as an accepted result.
