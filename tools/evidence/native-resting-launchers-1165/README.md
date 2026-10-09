# Native resting human launch and short-run evidence

Launch the current whole-body scene on the retained SSH Mac mini, with manual layer and camera controls:

    ssh macmini 'python3 /Users/n/numi-human-resting-evidence-20261005/native-manual-inspection-launcher-1165/launch_manual_1165.py'

The default is one finite 310-second native session. Each invocation creates a new output directory and calls the existing Human resting owner. Use --dry-run to check pins and print the command without starting a simulation; --seconds 20 selects the tested short smoke duration. This launcher uses the current 1159 anatomy and existing 018 viewer/014 physical runtime. It omits the presentation-only inspection tour so the native controls are no longer reset each frame. Actual mouse interaction has not been exercised in the locked remote console.

The separate launch_control_1159.py reproduces the pinned P17 control with its automatic recording tour and creates a fresh output directory. The registered baseline/intervention study remains independent; these launchers do not alter its invocation, source pins, or protocol. Both use existing owner CLI output and validation formats, with no alternate physical update path.

The manual 20-second smoke exited successfully with the exact 25 asset hashes and loaded runtime of the recording preflight. Its 1,250 rows and 60 physiological fields were byte-identical to that control. The retained movie has 315 image frames matching 315 surface audit rows; initial, middle and terminal frames show Skin / Whole body with advancing physiological values. The AVFoundation reader checks sorted presentation timestamps and sampled frames; it does not certify original compressed decode-order continuity.

The separate full 1120/1141 native geometry checks passed all eight accepted preflight poses: zero unallowed lung/diaphragm contacts, zero degenerate faces, and zero changed-face skin intersections. All 56 lung/diaphragm owner-pose comparisons also matched the production-kernel predictions exactly. These are discrete 20-second checks, not proof of continuous-time clearance or five-minute endurance. The registered 310-second baseline and intervention runs are still pending at this commit.

The compact reports and original file hashes are included here. Large anatomy assets, complete movies, failed attempts and raw traces remain in the pinned Mac-mini evidence tree. Source credits and mixed-source reference-adult limitations are in ATTRIBUTION.md; this is not anatomy measured from one person or clinical validation.
