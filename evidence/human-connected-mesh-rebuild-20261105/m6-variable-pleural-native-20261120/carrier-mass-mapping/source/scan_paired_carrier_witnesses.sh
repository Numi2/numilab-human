#!/bin/bash
set -euo pipefail
catalog=${PWD%/*}/captured-fat-sliver-495/proposal/compiler/mass-support
awk 'NF!=2 || ($1!="C" && $1!="D") {exit 1} END {if(NR==0) exit 1}' fine-paired-route-targets.txt
xcrun clang++ -std=c++20 -O3 extract_paired_carrier_witnesses.cpp -o extract_paired_carrier_witnesses-fast
for role in skin fat; do
    source="$catalog/$role-carrier-witnesses.csv.gz"
    case "$role" in
        skin) sha=a96f2ee1693a716fc633e70841f35556b4b6a0ce675fcd0c5af69b3de7090a98 ;;
        fat) sha=7ae070d92af8fc986dc7c0c932c11de73ba1d96694d349b2a6560db456fbc492 ;;
    esac
    shasum -a 256 -c <<<"$sha  $source"
    /usr/bin/gzip -dc "$source" | ./extract_paired_carrier_witnesses-fast fine-paired-route-targets.txt > "fine-paired-$role-carrier-witnesses.csv"
    wc -l "fine-paired-$role-carrier-witnesses.csv"
done
