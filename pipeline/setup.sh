#!/bin/bash
mkdir -p pipeline/input/rt/

curl https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst -o pipeline/input/metadata.tsv.zst
unzstd pipeline/input/metadata.tsv.zst -o pipeline/input/metadata.tsv

poetry run python -m pipeline.simulate_rt
