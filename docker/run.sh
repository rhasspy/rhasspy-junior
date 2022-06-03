#!/usr/bin/env bash

cd /app

echo 'Training'
scripts/train.sh \
    --config junior.toml "$@"

echo 'Training complete. Running'
scripts/run.sh \
    --config junior.toml "$@"
