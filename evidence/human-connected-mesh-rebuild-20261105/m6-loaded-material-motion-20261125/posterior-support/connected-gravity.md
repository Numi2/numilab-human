# Can the registered bed footprint support the skeleton?

The public launcher runs the full registered material owner. Two accepted steps reproduce a dynamic release with the skeleton accelerating toward the bed at 9.798 m/s². The native mass ledger accounts for 97.13 kg. The bed carries approximately 11 N after 15.625 µs. The current scene has yet to demonstrate supported rest.

## Through the connected material

The static preparation uses the native dermis and adipose stored energies, shared gluteal ports and the measured cortical contacts. A rigid donor translation loads the port and cortical branches together. All participating tissue nodes can move along the bed normal. There are no fixed interior cut surfaces. This calculation holds the transverse displacements, skeletal joint angles and cortical feature normals fixed. Its 96.656 kg gravitational submodel excludes approximately 0.476 kg belonging to other native tissues.

| Measured or calculated quantity | Result |
| --- | ---: |
| Coupled stiffness at the measured 16-point footprint | 1,214.07 N/m |
| Gluteal and cortical contributions | 841.72 and 372.35 N/m |
| Force closure error | 3.34 × 10⁻⁸ N/m |
| One-sided gravity solution | 255 loaded rows, 948.193 N |
| Skeletal descent demanded by that linearization | 359.28 mm |
| Admissible fraction after testing all 1,457,778 cells | 2⁻²⁵ |
| Skeletal descent at that fraction | 10.7 nm |
| Skeletal force imbalance, before and after | −810.00694 N and −810.00692 N |

Deep fat cell 784193 begins at a shape measure of 1.0000848 × 10⁻⁵, against the existing 1 × 10⁻⁵ material limit. The solver did not install this proposal in the scene. Balancing external force in the linearization leaves the finite material configuration and individual body reactions unqualified.

The earlier frozen-direction estimates treat all rows as unilateral. Native contacts include 14 bilateral vector springs. With those directions free and the 2,199 unilateral directions fixed, the optimistic peak extension or closure required in the current contact set is 411 mm for joint-stops and 402 mm for bilateral heels. These are conditional lower bounds. A changed contact footprint changes the calculation.

On SSH Numi, run from the Human repo with RUN set to the retained full joint-stops capture. The preparation uses Accelerate sparse Cholesky and the existing Python environment. The native simulation continues to use Metal.

```sh
v=tools/native_ligament_contact/registered_material_volume
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
"$py" "$v/regional_bed/solve_connected_normal_transfer.py" "$RUN" gravity.json --gravity-proposal gravity-field.npz
```

## Native forearm comparison

A left forearm exterior node contributes an 829.87 N bed-normal material energy gradient at opening. This is internal prestress. Relaxing the 11 connected surface vertices with all 123 incident tetrahedra, cortical attachments and surrounding tissue surfaces accounted for reduced the patch energy from 0.30078 to 0.26573 J. Maximum surface movement was 0.423 mm. The first trial introduced three dermal strain failures after release. Constraining the next starting shape with the measured trial displacements cleared those three regressions. All native dynamics were integrated again from initialization.

At 15.625 µs, the accepted patch has 791 intersecting surface pairs versus 793 in the matched control, with two removed and none introduced. Across all four physical walls, the same 6,568 transverse cortical pairs remain. There are 40 low-shape, 506 principal-stretch and seven volume-ratio failures. The native step certificate and the stricter anatomical material report answer different questions. The public material verifier exits with status 1 on this capture.

The forearm comparison carries 10.98 N on the bed. Its end-to-end runtime was 232.77 seconds on the Apple M6. Neither of the two compared physical profiles qualifies as settled or for extended motion. The new material is available as an explicit comparison profile. The public default retains the joint-stops control.

```sh
bash scripts/run_resting_human_on_numi.sh --dense --posterior-support forearm-material --steps 2
```

[Matched native measurements](forearm-material-motion.json) · [Verified recovery of native owners, input changes, captures, failed trials and source](forearm-material-recovery.json)
