# Exact joint source-solid boundary certificate

## Scope and result

This package records exact source-surface evidence for the paired fine skin and fat boundary. It does not qualify the coarse physical execution mesh. No O, I, D, or back source geometry was modified for these audits. No device simulation was created and no physical step was accepted.

The frozen inputs are the closed skin surface (SHA-256 352dc5e1ebfeca7f8c7dc13075dbcf0abc8d15a842f52057ba7744f998d5fa23) and closed fat surface (b02a064f088e625dfb11c98fd2859e13bd271224bd13d050c7b848c217d856b2). Their reports and launch manifests are included. The shared I interface has 205,239 bit-identical triangles with exactly opposite orientation. The full exact cross-surface audit found zero pairs for O-I, O-D, skin-rim-D, fat-rim-O, skin-rim-complete O/I, fat-rim-complete I/D, D-I, and the cross-rim check. All closed boundaries are oriented and edge-closed.

Run 007 authenticated the complete closed-surface component partition against the fat source rows. It classified all 91 detached fat components outside the closed skin, while retaining both negative cavity shells and their signs. Run 012 revalidated that component mapping, then located a representative point for each complete nonshared boundary component: the skin O-plus-rim component and all 92 reversed-D-plus-rim fat components are outside the opposing closed solid. Together with the zero exact cross-boundary pairs and oppositely oriented common I interface, this is the exact source-boundary evidence for no common open material interior under the reported closed-oriented-solid interpretation.

## Retained Boolean contradiction

The prior Manifold Boolean result remains a failure: Error.NoError with positive reported intersection volume 1.0353479473180264e-14 m3. Its localized diagnostic identifies a 4-face, 4-vertex tetrahedral component of about 1.0353422257115748e-14 m3. This package does not change that numeric result, set it to zero, waive the zero-volume gate, or alter geometry. Exact boundary evidence and the positive Boolean diagnostic remain together for review; the numerical contradiction is unresolved.

## Failed preparation attempts

Runs 008-011 are preserved with launch pins and stdout/stderr. They stopped on setup or report-shape errors: missing closed-skin pin (008), strict fat face-layout mismatch (009), malformed skin source-role map (010), and a location-field nesting error after index preparation (011). They are tooling failures, not geometry passes or failures. Run 012 is the corrected completed certificate.

## Reproduction and limits

The reusable scripts and exact inputs/hashes are in the tools and evidence paths. Run manifests record the SSH Numi host and serial wrapper. The run 006 full Boolean intersection NPZ is intentionally not copied; the compact Boolean stage, report, exact unit parity, and failure record are retained. The separate run 656 localized report and its small audit script are included; its full intersection NPZ is omitted. The component maps in runs 007 and 012 preserve source-face row membership and classification.

This is a source-solid boundary certificate only. The physical execution mesh still has unresolved admission gates. Runtime created: 0. Accepted physical steps: 0. No physical, motion, performance, or clinical qualification is claimed here.
