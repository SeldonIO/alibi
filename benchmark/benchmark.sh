#!/bin/bash -e

# Parse arguments
for ARGUMENT in "$@"; do
  KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
  VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)

  case "$KEY" in
  VERSION_1) VERSION_1=${VALUE} ;;
  VERSION_2) VERSION_2=${VALUE} ;;
  EXPERIMENT_1) EXPERIMENT_1=${VALUE} ;;
  EXPERIMENT_2) EXPERIMENT_2=${VALUE} ;;
  *) ;;
  esac
done

echo "VERSION_1: $VERSION_1"
echo "EXPERIMENT_1: $EXPERIMENT_1"
echo "VERSION_2: $VERSION_2"
echo "EXPERIMENT_2: $EXPERIMENT_2"

checkout_error() {
  BRANCH=$1
  echo "Could not checkout $BRANCH"
  echo "Ensure the branch or commit hash exist and all changes on current branch are committed!!!"
  echo "Exiting"
} >&2

experiment_error() {
  BRANCH=$1
  YAML=$2
  echo "Could not run expermient on $BRANCH"
  echo "Ensure the configuration $YAML exists in benchmark/configs/ or check your expermient code!"
  echo "Exiting ..."
} >&2

echo "Checking out ${VERSION_1}..."
git checkout "$VERSION_1" || {
  checkout_error "$VERSION_1"
  exit 1
}

HASH="$(git rev-parse HEAD)"
echo "Runnig expermient with configuration ${EXPERIMENT_1} on this branch ..."
python benchmark/experiment.py --config "benchmark/configs/$EXPERIMENT_1" --hash "$HASH" ||
  {
    experiment_error "$VERSION_1" "$EXPERIMENT_1"
    exit 1
  }

echo "Checking out ${VERSION_2}..."
git checkout "$VERSION_2" || {
  checkout_error "$VERSION_2"
  exit 1
}

HASH="$(git rev-parse HEAD)"
python benchmark/experiment.py --config "benchmark/configs/$EXPERIMENT_2" --hash "$HASH" ||
  {
    experiment_error "$VERSION_2" "$EXPERIMENT_2"
    exit 1
  }
echo "Benchmarking complete!"
