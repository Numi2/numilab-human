# Rejected back geometry on Apple M6

The back455 candidate failed the existing native whole-body surface audit at initialization. The run exited 1 after 8.38464125 wall seconds, before any physical step advanced. It is not an accepted anatomy repair. The previously retained back408/right177 60-second scene remains the working baseline.

Source-space Float32 nondegeneracy did not survive the existing GPU parent transform: the rejected pack contains 732 exactly zero-area row153 faces. Face4040 has two source vertices separated by 7.45nm that become identical in the rendered Float32 coordinates. The native whole-scene audit reported 1,545 invalid faces; this is a different scope from the independent row153 count.

The copied audit is read-only and pins the rejected native pack, source patch and preparation. The pack remains at the retained remote path; it is not duplicated here. The source-space storage guard published in cad55142 is still useful, but does not qualify transformed geometry.

A subsequent CPU default-pose predictor also rejects known native-safe back408 row157. Therefore CPU prediction is a diagnostic, not a replacement for the actual Metal transform and native audit. New repairs must preserve interface topology and source ancestry, and pass native checks over the breathing cycle before admission.

No performance, complete anatomy, or five-minute qualification claim is made.
