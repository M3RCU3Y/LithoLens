#!/usr/bin/env bash
set -euo pipefail

mkdir -p references
cd references

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "Already cloned: $dir"
  else
    git clone "$url" "$dir"
  fi
}

clone_if_missing https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition.git force-2020-official
clone_if_missing https://github.com/artem-potlog/well-log-facies-prediction.git uncertainty-facies
clone_if_missing https://github.com/mycarta/Force-2020-Machine-Learning-competition_predict-lithology-EDA.git force-2020-eda
clone_if_missing https://github.com/olawaleibrahim/2020_FORCE_Lithology_Prediction.git force-2020-solution
