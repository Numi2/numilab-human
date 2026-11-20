#!/bin/bash
set -euo pipefail
R=$PWD
C=${R%/*}/captured-fat-sliver-495/proposal/compiler
cd "$R"
for role in skin fat; do
    witness="$C/mass-support/$role-carrier-witnesses.csv.gz"
    case "$role" in
        skin) expected=a96f2ee1693a716fc633e70841f35556b4b6a0ce675fcd0c5af69b3de7090a98 ;;
        fat) expected=7ae070d92af8fc986dc7c0c932c11de73ba1d96694d349b2a6560db456fbc492 ;;
    esac
    shasum -a 256 -c <<<"$expected  $witness"
    /usr/bin/gzip -dc "$witness" | ./extract_mass_alternate_witnesses mass-cut-witness-targets.txt > "fine-$role-cut-witnesses.csv"
    wc -l "fine-$role-cut-witnesses.csv"
done
