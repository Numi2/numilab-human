# Left FDP5 enthesis connected-component recovery

The left flexor digitorum profundus to digit five (`FDP5_l`) insertion now
uses a distributed four-node `NHTENDON3` envelope on the named BodyParts3D
distal phalanx member `FJ3184`. The authored MyoSim route point, rigid-body
ownership, force authority, and all existing admission gates remain unchanged.

## Defect and correction

`FJ3184` contains 198 triangles split across six disconnected components. The
old search stopped after its globally nearest triangle, a detached three-vertex
scan fragment. That fragment cannot provide a conditioned four-node patch, so
the insertion remained an exact source-point fallback even though the main
190-triangle anatomical component satisfies every existing gate.

The compiler now retries deterministic connected components only after the
nearest component fails patch construction or conditioning. Components are
ranked by exact point-to-surface distance and source triangle index. A candidate
must still satisfy the same 12 mm distance and radius gates, force and moment
conservation gates, and sampled total-force amplification limit. Original mesh
triangle and vertex provenance is retained in the manifest.

For `FDP5_l`, the accepted main component has:

- surface distance: `5.13640549 mm`
- patch radius: `7.96028079 mm`
- sampled total-force amplification: `3.11447631`
- force residual: `1.09e-15`
- moment residual: `4.26e-18 m`
- endpoint migration: `0`

The same fallback was tested against every prior conditioning failure. Only
`FDP5_l` becomes admissible, so the change does not relax or bypass the other
fail-closed endpoint decisions.

## Runtime qualification

The rebuilt payload contains all 832 endpoints: 639 distributed envelopes and
193 exact point laws. Of the envelopes, 629 terminate on registered BodyParts3D
bones. The new payload passed the reference CPU/Metal transfer probe with Metal
API validation enabled on Apple M4 Pro. The Metal pass executed all 832
transfers with maximum force residual `0.000246159 N`, maximum moment residual
`7.17526e-6 N m`, maximum generalized correction `0.000732422`, and
byte-identical tendon replay.

A separate two-step, `0.1 ms` selected-control transaction drove source muscle
309 (`FDP5_l`) with a `0.2` activation increment. It executed 1,664 endpoint
transfers, including 1,278 envelope transfers. Maximum force and moment
residuals were `0.000126749 N` and `1.23635e-6 N m`. The distributed transfer
was a borrowed consumer of the source-route `J^T` force in the same Metal
command buffer; direct rigid-state effect remained bitwise absent, injected
consumer rejection preserved the result, and replay was bitwise.

The machine-readable receipt is
[`m4-pro.json`](media/numi-human-hand-enthesis-component-v1/m4-pro.json).

## Reproduction

```sh
PYTHONPATH=src python3 -m numilab_human.cli \
  numi-human-tendon-envelope-payload \
  --artifact Build/nheq1 \
  --bone-artifact Build/lower-interface-v3-bones \
  --output Build/hand-enthesis-component-v1-tendon \
  --migrate-semantic-rigid-foot-endpoints
```

The runtime transaction uses the generated `NHTENDON3` payload with
`--selected-tendon-control`, `--activated-source-muscle-index 309`, the current
support-contact payload, and the current `NHEQ1` joint-equality payload.

## Evidence boundary

This closes one incorrect point fallback and makes the left fifth deep flexor
insertion a conservative distributed surface transfer. It does not validate
finger contact, independent phalanx articulation, tendon wrapping through the
carpal tunnel and pulleys, deformable tendon/fascia material, or the remaining
193 point laws. Those remain separate mechanics gates.
