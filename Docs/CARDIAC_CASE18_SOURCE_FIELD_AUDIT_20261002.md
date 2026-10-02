# Rodero case18 source-field audit — 2 October 2026

The complete pinned case18 archive was checked against the source configuration
before treating its anatomy and electrical inputs as source data. Its SHA-256 is
b50919a711dc3914cc0a4dd9cabc19679a9f36be9ac6eb26a23ca6799f501720, matching
the retained source identity. The archive contains exactly one regular member,
18.vtk, with SHA-256
f0d4f3fc21ea229fa1b334888d374ee606a2915b414fff047764aad44f1c0a41.

The VTK declares 300,965 points and 1,470,083 tetrahedra. Its only data fields
are the cell material ID, ventricular fibres and sheets, and the four UVC fields
rho, phi, Z and V. It contains no case18 activation-time field, voltage/ionic
state, or fast-endocardial cell tag. The official [Zenodo record](https://zenodo.org/records/4590294)
describes the per-case meshes and simulation outputs; this audit inspected the
actual case18 member, rather than inferring its contents from the general
dataset description.

This closes the uncertainty about the contents of the pinned case18 archive.
It does not close electrical reproduction: the published activation metrics
remain aggregate simulation outputs, while the current activation field and
fast-layer membership are reconstructed inputs. Do not tune the missing layer
to the aggregate timing metrics and then present the fitted result as an
independent reproduction. Continue to label that layer as a hypothesis until a
source case18 tag or local activation reference is obtained. Native accepted
electrical steps, voltage/ionic dynamics, and electrical-to-mechanical coupling
remain open.

The executable [audit](../tools/audit_cardiac_case18_source_fields.py) checks
the archive and member hashes, the one-member boundary, mesh section counts,
and the complete VTK declaration list. It fails closed if an electrical,
fast-layer, or other undeclared field appears. The [receipt](media/cardiac-case18-source-field-audit-20261002/receipt.json)
records the exact inventory and claim boundary.

Rerun it against a copy of the pinned archive:

    .venv-mujoco312/bin/python tools/audit_cardiac_case18_source_fields.py \
      --archive /path/to/18.tar.gz \
      --output Docs/media/cardiac-case18-source-field-audit-20261002/receipt.json
    .venv-mujoco312/bin/python -m pytest -q \
      tests/test_cardiac_case18_source_fields.py

The source activation reconstruction now links this inspection as its provenance
gate. Its measured 3.1997 ms LV activation-span and 7.4885 ms LV 10–90% timing
differences from the published case18 simulation remain unresolved.
