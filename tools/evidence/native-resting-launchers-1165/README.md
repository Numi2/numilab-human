# Native resting human launch and retained evidence

Launch the whole-body scene on the SSH Mac mini, with manual layer and camera controls:

    ssh macmini 'python3 /Users/n/numi-human-retained-delivery-20261009/launchers/launch_manual.py'

Run one native simulation at a time. The default is a finite 310-second session. Each invocation creates a fresh output directory and calls the existing Human resting owner. --dry-run checks pins and prints the command without launching; --seconds 20 selects the previously tested smoke duration. The launcher uses 1159 anatomy, viewer 018 and physical runtime 014. It omits the presentation tour so native layer and camera controls are not reset each frame. Mouse interaction has not been exercised in the locked remote console.

The recorded-control launcher reproduces the current P18 control with its automatic layer tour:

    ssh macmini 'python3 /Users/n/numi-human-retained-delivery-20261009/launchers/launch_recorded_control.py'

Both launchers use the existing owner CLI and output formats. The recorded-control launcher pins Lab owner f4f1d1d, which verifies the exact final accepted capture rather than rejecting its additional terminal frame. Its actual 20-second smoke completed successfully, preserved the prior physiology CSV byte-for-byte, and verified accepted step 10000 at 20.000000949949026 seconds. The updated launchers also pass pin-checking dry runs. The files retain their older names in this publication directory for link compatibility.

The fresh P18 two-arm study is registered at /Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170. Both arms completed 310.000014724 simulated seconds, including 300 seconds after initialization. The registered CO2 response, ventilation direction, and late recovery checks passed. However, full late-pose audits found skin and lung intersections and small persistent COM drift remains unexplained; the launcher is not final anatomical acceptance. Exact completion, performance, recordings, failures and ongoing repairs are recorded in [the closed 1170 evidence](../native-integrated-resting-1170/README.md). The fixed 310-second protocol retains 10 seconds of initialization and a temporary half-drive interval at [60,100) seconds.

Earlier manual inspection completed a 20-second smoke with 1,250 rows and 60 physiological fields matching the recording preflight. Its retained compact recording review describes 315 frames matching the surface rows, with whole-body Skin frames at the start, middle and end. The AVFoundation review checks sorted presentation timestamps and sampled frames, not original compressed decode-order continuity.

Earlier 1120/1141 geometry reviews reported eight accepted short-run poses with zero unallowed lung/diaphragm contacts, zero degenerate faces, and zero changed-face skin intersections. All 56 lung/diaphragm owner-pose comparisons matched production-kernel predictions. These are discrete 20-second results. A separate storage cleanup subsequently removed the raw 1120 report and the original 917 trial, so their published compact summaries are historical evidence, not substitutes for the new long runs. The retained deletion incident records the missing files; failed attempts have not been promoted to successful receipts.

The closed study uses a newly executed and verified 1173 full-q/terminal reference. Its measured 1174 comparison to the current scene found 75,000 sampled physiology values and eight body captures equal. The original 931 metadata remains unavailable; the new metadata is identified separately. Both movies decoded completely with 4,846 frames each and all seven anatomical layers. Physiology analysis is complete; both lung audits and both complete skin audits expose the documented failures. Skin and lung corrections continue under separate candidate identities.

Source credits and mixed-source reference-adult limits are in ATTRIBUTION.md. This is not anatomy measured from one individual or clinical validation. Large current assets and evidence stay on the Mac mini; closed current inputs and recordings carry reversible user-immutable flags to protect them against accidental cleanup.
