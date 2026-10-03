#!/usr/bin/env bash
# Download the datasets used by the examples into examples/data/.
#   UCR ECG5000 (uDTW forecasting)      ~10 MB  timeseriesclassification.com
#   NW-UCLA Multiview skeletons (JEANIE) ~14 MB  all_sqe.zip as distributed with CTR-GCN
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data

if [ ! -f ECG5000_TRAIN.txt ]; then
  curl -fsSL -o ECG5000.zip https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip
  unzip -q -o ECG5000.zip && rm ECG5000.zip
fi

if [ ! -d nwucla/all_sqe ]; then
  curl -fsSL -o nwucla_all_sqe.zip "https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=1"
  unzip -q -o nwucla_all_sqe.zip -d nwucla && rm nwucla_all_sqe.zip
fi
echo "datasets ready in $(pwd)"
