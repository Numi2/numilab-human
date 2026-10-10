# Shared toe-joint passive restraint: bounded native sensitivity

The supported reference body previously settled both shared toe coordinates onto their -30 degree stops. An opt-in aggregate passive spring now uses the existing GPU implicit passive-joint program for those two coordinates. The omitted option preserves prior behavior. No default stiffness is selected by this evidence.

Four identical 16.00000076 simulated-second runs on the SSH Apple M4 Pro Mac mini varied only stiffness (0, 0.5, 1, 2 N m/rad). All used the same 72 kg body, flat support plane, 26-surface anatomy composition, dynamic release, contact settings, and native integrated respiratory/cardiac/gas-exchange owner.

| Stiffness N m/rad | Right/left mean toe angle, degrees (14–16 s) | Terminal forefoot–toe bone triangle intersections | Native real-time factor |
|---|---|---|---|
| 0 | -30.000 / -30.000 | 295 | 0.10924 |
| 0.5 | -27.954 / -30.000 | 286 | 0.10988 |
| 1 | -12.972 / -14.139 | 78 | 0.11035 |
| 2 | -6.086 / -6.566 | 77 | 0.11052 |

At 1 and 2 N m/rad both coordinates had zero joint-limit impulse throughout the 1,000 observed final steps. At zero both remained on their stops; at 0.5 the left remained there. This demonstrates a missing passive-restraint contribution, not completed foot anatomy. Residual bone crossings and tendon self-intersections remain. Triangle-pair counts are not penetration depth. Restoring torques in the report are calculated offline as -Kq, not a GPU force readback.

## Model and provenance

The source articulation has one toe coordinate per foot shared by five rays (q111/v110 and q125/v124). Rest is zero radians. K is a declared reference aggregate; it is not five independently measured joints, a clinical fit, or a muscle actuator. Model mapping is checked on load. Positive K rejects incompatible passive removal and unbound static recruitment caches. The force/state owner remains the existing Metal passive-joint solver.

[Heng et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5080733/) report first-MTP passive dorsiflexion quasi-stiffness in 13 healthy participants; the experienced examiner's session-one mean 14.9 N mm/degree converts to about 0.854 N m/rad. That provides a scale for this sensitivity sweep, not validation of an aggregate five-ray law. Measurement repeatability and anatomical scope limit that comparison.

## Verification and retained execution

Every run completed with unchanged pinned inputs, verified loaded Metal runtime, finite coupled traces, matching body/physiology clocks, and zero root assistance. Native rejected-step/accepted-prefix/retry checks passed; exact lines are retained per arm. Each continuous native movie contains 252 frames and five timing markers; the existing AVFoundation inspector checked frame/trace alignment and monotonic timestamps and extracted layer views. Movie paths and SHA-256 identities remain in summary.json on the Mac mini.

K=0 reproduced all 157 body poses and all 860 captured anatomical surfaces exactly at steps 0 and 8000 versus the prior 26-surface run. Coupled physiological fields matched; 125 diagnostic rows differ in contact/residual fields because the final q-observer interval changed from eight-step to per-step diagnostics. This is not claimed as a byte-identical full trace.

The Human launcher tests passed (31 tests plus 36 subtests), passive-joint CTest passed, and both native and generic visual-probe targets built. Sources were held fixed throughout the runs: pre-commit revisions plus exact source patches, binary/library/shader hashes, configuration and commands are in each run-declaration and execution record. The package preserves compact q/limit traces, losslessly compressed full coupled traces, rejection evidence, and hash-bound geometry audits.

To reproduce an arm, use its run-declaration's immutable inputs and execution.json command on the Mac mini. The retained prepare_mtp_native_sensitivity_001.py records assembly, and package_mtp_native_evidence_004.py records this package extraction. These are existing native-smoke/accepted-state formats, not an additional simulation implementation.

This is a 16-second body-mechanics increment. It does not qualify the complete anatomy, five-minute final-body demonstration, all physiological ranges, or real-time performance. Native RTF excludes launcher setup; launcher timings are also retained.
