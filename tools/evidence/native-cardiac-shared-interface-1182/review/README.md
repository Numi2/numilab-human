# Cardiac connected-patch audit review

This is a read-only code review of the optional classify_connected_patches=True branch in cardiac_partition_certificate.audit_shared_interface_partition.

Verdict: no blocking issue found. The branch preserves the exact contact predicate and fail-closed geometry checks. It computes region membership once per edge-connected uncut boundary patch, after blocking every declared interface edge and each exact participant edge touched by an admitted contact. An exact centroid parity query classifies each patch; boundary or indeterminate points are rejected. The existing slower per-face classification remains the default.

The diff under review is bound by review.json and retained in reviewed-diff.patch. The focused certificate tests pass under the supported Python 3.13.13 owner environment (18 tests). The default Mac Python 3.9.6 run has one interpreter-compatibility error in the pre-existing test_hex_encoding_roundtrips_large_exact_moments_without_global_changes, because sys.get_int_max_str_digits does not exist in Python 3.9; this is recorded verbatim and is not introduced by the patch.

The certificate still explicitly reports biological_valve_interface=false and mechanical_mass_assigned=false; this review makes no biological or valve-dynamics claim.
