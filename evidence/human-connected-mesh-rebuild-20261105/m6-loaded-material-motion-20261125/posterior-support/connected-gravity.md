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
