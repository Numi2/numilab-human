# Rodero-2026 HCM1 source activation import

This import adds one reproducible, source-bound HCM activation map to the Human
tooling. The source is the HCM1 patient mesh and sample 53 reaction-eikonal
activation map from the authors' 2026 publication datasets. The HCM mesh and EP
archive are pinned by byte count, MD5, and SHA-256. The importer verifies the
legacy VTK point count, sample-specific EP parameters and source tags, and
requires one finite activation value for every VTK point. It preserves the
publisher's `-1` inactive-point sentinel as a sentinel rather than treating it
as a time.

Point identity follows the publisher's own
[`visualise_EP.py`](https://github.com/CEMRG-publications/Rodero_2026_JMCC/blob/0e13e424ceb76967ae8341a9419cf0cf3525600b/simulation_toolbox/visualise_EP.py):
it reads the `.dat` vector with `numpy.loadtxt` and assigns the vector directly
to point data on the corresponding VTK mesh. The pinned script labels the
activation progression in milliseconds and moves negative entries beyond the
active range only for visualization. This importer preserves the source `-1`
sentinels verbatim. No interpolation or point reordering is performed. The
imported little-endian float64 file follows the same 749,238-point order as
`HCM1.vtk`.

## Reproduce

Download `HCM1.vtk` from the [HCM four-chamber mesh record](https://zenodo.org/records/21282274)
and `HCM1_EP_light.tar.zst` from the
[HCM electrophysiology record](https://zenodo.org/records/21720235), then run:

```sh
uv run --python 3.11 --extra cardiac-electrical -- numilab-human rodero-hcm1-source-activation \
  --mesh-vtk /path/to/HCM1.vtk \
  --ep-archive /path/to/HCM1_EP_light.tar.zst \
  --output-dir Build/cardiac-rodero26-hcm1-activation
```

The command writes `hcm1-sample53-activation-time-ms.f64le` and an immutable
`source-activation-import.json` receipt. Running it again against the same
inputs verifies and reuses identical outputs; changed inputs or outputs fail
closed.

## Evidence boundary

This imports one published HCM patient variant, not the Healthy case18 source.
It preserves a spatially ordered activation-time field from an upstream
reaction-eikonal workflow. It does not run electrical dynamics in Numi, carry
voltage or ionic state, register HCM1 to the current Human heart, couple
electrical activation to mechanics, model whole-body circulation, or qualify a
heartbeat or clinical prediction. Those remain open integration and
qualification tasks.
