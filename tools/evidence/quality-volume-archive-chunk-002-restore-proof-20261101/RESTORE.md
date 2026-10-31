# Retrieve and restore an archived proposal

The release asset is hosted outside Git: [chunk 002 release](https://github.com/Numi2/numilab-human/releases/tag/human-quality-volume-archive-20261031-chunk-002). Download it and check the whole-archive SHA before reading members:

```sh
curl -fL https://github.com/Numi2/numilab-human/releases/download/human-quality-volume-archive-20261031-chunk-002/quality-volume-archive-chunk-002.tar.zst -o quality-volume-archive-chunk-002.tar.zst
printf '%s  %s\n' 51ed5f418bda9640a3204f997219c0416d50dccb13bbf0c9f647f6ff37e8e3ce quality-volume-archive-chunk-002.tar.zst | shasum -a 256 -c -
```

To restore the exact sample used in the retained restore smoke, create a new destination and extract its content-addressed blob. `set -o noclobber` prevents overwriting an existing output.

```sh
set -o pipefail
set -o noclobber
mkdir -p restored
zstd -dc quality-volume-archive-chunk-002.tar.zst | tar -xOf - blobs/99224b13ad7d6b45be80f9c2cf075b817186b2a8cebf3d41a6fe956dcd5bacd9 > restored/fat-fixed-boundary-quality-queue.npz
printf '%s  %s\n' 99224b13ad7d6b45be80f9c2cf075b817186b2a8cebf3d41a6fe956dcd5bacd9 restored/fat-fixed-boundary-quality-queue.npz | shasum -a 256 -c -
```

The sample is indexed in `payload/plan/archive-index.json` with its original relative path and size. For another indexed blob, select its SHA-256 from `unique_blobs`, extract `blobs/<sha256>`, and compare the result to the indexed size and digest. Archived proposals are historical inputs, not current geometry.
