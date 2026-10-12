# Three back cap flips verified in the native whole-body scene

Three inferred cap retriangulations replace six face slots in rows 157/158. Exact source coordinates and raw parent maps equal the existing back408 reference. The executable tools/replay_back_cap_flips.py replays the three pinned reports, checks exact planar patch boundary/orientation, rejects measured-source-face edits, and verifies the resulting geometry against candidate412. Its real-asset replay passed.

On SSH numi (Apple M6), native scene back-forearm-native-014-016-back412-right177 completed 16 simulated seconds in 106.249120125 wall seconds, approximately 0.151x realtime. No input changed. This is a bounded geometry increment, not the full deliverable or a performance qualification.

All eight captures preserve exact vertex motion for the changed rows, all 872 other inspected surfaces, all 157 physical body poses, and the complete physiology trace compared with back-forearm-native-014-013-right177. No exact collapsed faces occur in the changed rows; the native whole-mesh gate passed. Normals were recomputed for the new faces. Shared interface cardinalities remain unchanged (2,792 paired faces).

At steps 0/8000, exact raw self-contact pairs drop from 47/46 to 42/34 in row157, and 31/36 to 27/32 in row158. Residual contacts remain. Changed-face neighbor checks find no new target surfaces or target triangle IDs at either endpoint, but raw pair counts on existing interfaces vary with retriangulation. This does not establish whole-body clearance.

The continuous native recording has 252 image frames, 5 timing markers, 100.445 wall seconds, maximum image gap 1.72333 seconds. Its anatomical layer captures were extracted; the muscle frame was visually inspected. frame-muscles.png is an actual native frame. The recording and large assets remain on numi at the paths pinned in source-pins.json.

Native launch:

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/back-forearm-native-014-016-back412-right177/launch_arm.py baseline

This retained run directory already exists; preserve it and prepare a fresh output for reruns.

Limitations: the left forearm/PQ crossings and broader back-interface defects are unresolved. Broad back455 failed native initialization and its successors remain unadmitted. The left source-owner harmonic field also failed late-pose self-intersection checks and was not installed. The prior 60-second baseline remains retained. No five-minute final-anatomy or clinical validation claim is made.

An initial audit incorrectly required normal bytes to be unchanged along with positions. It stopped before capture checks. The corrected audit permits recomputed normals in the two changed rows while requiring exact source positions/bindings and actual native positions. This diagnostic error and its log remain in the remote scene.
