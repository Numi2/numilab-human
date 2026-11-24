# Where the local posterior tissue solve carries load

The original preload was replayed on SSH Numi with the registered material, bone binding tables and gravity. Every original nonlinear iteration, contact gap, residual and termination value matched the retained result exactly. A force observer summed the existing constitutive corner forces at the fixed nodes after equilibrium. Subtracting the first converged state cancels the fixed gravity loading and opening internal forces. All vectors in the evidence include coordinates and signs. Positive normal force denotes the bed load carried away from the free tissue.

The patch contains 7,881 tetrahedra, 791 free nodes and 1,597 fixed nodes. Across the whole registration there are 14 explicit bonds. The patch contains zero. Of its fixed nodes, 322 appear in the native cortical contact sample table. Their inclusion makes them candidate contacts; the offline calculation fixes them along with the remainder of its cut boundary.

| Increment from 0.1% to 5% body weight | Normal force |
|---|---:|
| Increased bed load | 46.690214 N |
| Fixed nodes coinciding with explicit skeletal bonds | 0.000000 N |
| Fixed nodes eligible for native cortical contact | 0.002456 N |
| Other fixed tissue nodes | 46.687758 N |
| Magnitude of the force closure residual | 3.62e-08 N |

The local calculation attributed 99.9947% of the added vertical reaction to those other fixed nodes. Their neighbors can transmit physical load in the complete body. Their reaction and displacement must be obtained from that surrounding tissue and its actual skeletal interfaces to use this calculation for whole-body initialization. Rigidly fixing the cut supplies an external reaction. At the attempted 12.5% load the prior solver lost a positive contact tangent before equilibrium.

Use the fixed-boundary calculation for local constitutive and numerical comparison. Whole-body support requires a common accepted state with bed pressure, equal-and-opposite interface forces, skeletal reaction, weight and moment balance. The corresponding live 44-region contact and accepted-state measurements are in the adjacent registered-bed and full-volume evidence.
