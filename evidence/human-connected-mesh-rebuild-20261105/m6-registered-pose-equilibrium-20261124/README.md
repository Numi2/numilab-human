# Connected material registration under load

The reference material, source mass, outer skin, deep fat, cortical contact nodes, skeletal attachments and bed remained fixed. The offline fit minimized the existing dermal and adipose constitutive energy around 313 distorted tetrahedra. It moved 1,021 free material coordinates, including 852 on the common skin/fat interface. The native scene reads the resulting pose once at initialization. No relaxation or auxiliary support force runs in the physical timestep.

Stored elastic energy across 22,650 affected tetrahedra fell from 4.015 J to 2.640 J. At the left-ulna witness the constitutive force magnitude fell from 481.39 N to 56.36 N. All 1,457,313 native Float32 determinant intervals passed. The four material surfaces had zero crossing pairs at initialization. An earlier fit failed ten determinant intervals. Its failures remain available in the evidence.

The full variable-pleural-pressure scene rejected the first physical step with a global nonlinear residual of 0.0132492. As Newton progressed, some fat cells reached the material limit of J = 2.5. The final accepted line-search fraction for the attempted correction was 5.96 × 10⁻⁸ and the physical step rolled back. The control accepted its first step. The altered pose is an experimental negative result and stays off the launch path.

Static energy reduction alone is insufficient. The matched control has a captured first physical displacement that can be used to preserve the available deformation margin when fitting the initial pose. A new native release is still necessary to certify any such change.
