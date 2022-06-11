#!/usr/bin/env bash

cd /app

echo 'Training'
scripts/train.sh "$@"

echo 'Training complete. Running'
scripts/run.sh "$@"
