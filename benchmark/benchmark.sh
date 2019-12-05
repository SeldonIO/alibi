#!/bin/bash -e

# TODO: Handle config files missing
# Parse arguments
for ARGUMENT in "$@"; do
  KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
  VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)

  case "$KEY" in
  VERSION_1) VERSION_1=${VALUE} ;;
  VERSION_2) VERSION_2=${VALUE} ;;
  EXPERIMENT_1) EXPERIMENT_1=${VALUE} ;;
  EXPERIMENT_2) EXPERIMENT_2=${VALUE} ;;
  TYPE) TYPE=${VALUE} ;;
  *) ;;
  esac
done

# Use to check if config exists
CPATH=$(pwd)
set_default_config() {
  EXPERIMENT_1=$1
  TYPE=$2

  if [ -z ${EXPERIMENT_1} ]; then
    if [ -z ${TYPE+x} ]; then
      EXPERIMENT_1="default.yaml"
      EXPERIMENT_2="default.yaml"
    else
      echo "EXPLAINER TYPE: $TYPE"
      EXPERIMENT_1="default_$TYPE.yaml"
      EXPERIMENT_2="default_$TYPE.yaml"
    fi
  fi
}

# Set experiment to default value if not passed and get current branch name
set_default_config "$EXPERIMENT_1" "$TYPE"
if [ "$VERSION_1" = "this" ]; then VERSION_1=$(git rev-parse --abbrev-ref HEAD); fi
if [ "$VERSION_2" = "this" ]; then VERSION_2=$(git rev-parse --abbrev-ref HEAD); fi

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
EXPERIMENT_1_CFG="$CPATH/benchmark/configs/$EXPERIMENT_1"
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
EXPERIMENT_2_CFG="$CPATH/benchmark/configs/$EXPERIMENT_2"
python benchmark/experiment.py --config "benchmark/configs/$EXPERIMENT_2" --hash "$HASH" ||
  {
    experiment_error "$VERSION_2" "$EXPERIMENT_2"
    exit 1
  }
echo "Benchmarking complete!"
