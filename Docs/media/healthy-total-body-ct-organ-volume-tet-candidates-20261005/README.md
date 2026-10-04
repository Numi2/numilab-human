# Whole-body CT organ volume mesh candidates — 2026-10-05

This evidence records source-bound geometry increments for the 12 automatic organ and vessel labels in scans 001 and 002. Across the two scans, 7,350,007 source voxels are represented by 44,100,042 tetrahedra. Each scan has its own plan, source hashes, compiler receipt, VTK outputs, and independent verification.

The scan-001 flood-fill attempt failed at Lung and is retained in [`attempt-001`](attempt-001/); it filled a four-voxel enclosed void as solid. Direct NIfTI mask ingestion fixed the reconstruction method. Scan 001 is recorded in [`attempt-002`](attempt-002/), and the same registered pipeline now passes for [scan 002](scan-002/). Each [independent verifier](attempt-002/verify_candidate.py) rereads its exact scan from the registered archive and audits the serialized tetrahedra against each mask.

These remain automatic scan-specific segmentation geometry. They are not expert-reviewed anatomy, a Numi Human subject, physical organ ownership, calibrated tissue, vessel lumen/perfusion, mechanics, or physiological evidence.
