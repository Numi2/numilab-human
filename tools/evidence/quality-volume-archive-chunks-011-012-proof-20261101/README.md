# Quality volume archive chunks 011–012

This package preserves compact build plans and receipts for existing GitHub archive assets 011 and 012, plus an additive serializer-hash reconciliation. The archive payloads remain in the existing releases; no NPZ/NPY arrays or `.tar.zst` payloads are bundled here.

Both archive assets are independently recorded and were confirmed live as uploaded with matching asset IDs, SHA-256 digests, and sizes. Retained NPZ CRC/NPY-header readback and one-file restore receipts pass. The two index SHA fields differ by serialization: build receipts hash exact pretty JSON file bytes, while remote/readback receipts hash compact canonical JSON. Reconciliation-002 records both forms for 011–012; reconciliation-001 is preserved unchanged and remains scoped to 007–010.

The records cover 21 selected NPZ references / 21 unique blobs and 504,519,596 bytes of selected originals. These are archival/restore facts only; native steps are zero and no physical or current-geometry qualification is claimed. Chunk 012’s pre-retirement-proof erratum is retained unchanged.
