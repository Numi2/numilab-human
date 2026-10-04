# Attempt 001 — volume-sum gate failure

The source surface and voxel reconstruction reached tetrahedral mesh emission, but the independent volume closure gate failed because naive floating-point accumulation exceeded the preregistered tolerance. The partial mesh is retained for audit and was not admitted. The next attempt changes only the summation method to compensated accumulation; the source and acceptance tolerance remain fixed.
