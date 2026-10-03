# Human gap evidence reference audit

`numi human gap-execution` pins its registered documents, but its report does
not check the documents' local evidence links. The owner CLI now provides that
separate check with:

```sh
numi human gap-reference-audit --output <new-receipt-path>
```

To include the validated reference summary in the task/dependency report, run:

```sh
numi human gap-execution --validate-references --output <new-report-path>
```

The audit validates every path named by the execution registry, parses its
registered JSON and JSON.GZ files, checks inline Markdown links, image paths,
local heading anchors and linked-directory availability, and hashes each
resolved local file. It also pins the registry, direct evidence references and
the audit predicate source so the report is reproducible. It fails on unresolved
local targets or link syntax it cannot parse.

External URLs are listed but never fetched. A passing result verifies only that
the evidence-reference graph is locally navigable and syntactically readable;
it does not verify scientific claims inside a receipt or document, assess task
readiness, or qualify anatomy, physiology, contact, behavior, or runtime
performance. The authoritative completion ledger remains the status source.
