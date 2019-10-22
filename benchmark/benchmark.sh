#!/bin/bash

BRANCH_1=$1
EXPERIMENT_1=$2
BRANCH_2=$3
EXPERIMENT_2=$4

echo "Checking out ${BRANCH_1}..."
git checkout "$BRANCH_1"
echo "Runnig expermient with configuration ${EXPERIMENT_1} on this branch ..."
python benchmark/expermient.py --config "benchmark/configs/$EXPERIMENT_1"
echo "Checking out ${BRANCH_2}..."
git checkout "$BRANCH_2"
echo "Runnig expermient with configuration ${EXPERIMENT_2} on this branch ..."
python benchmark/expermient.py --config "benchmark/configs/$EXPERIMENT_2"
echo "Benchmarking complete!"