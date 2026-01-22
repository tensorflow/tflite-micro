#!/bin/bash
# tensorflow/lite/micro/tools/ci_build/docker_run_tflm.sh
# Helper to run a script inside a TFLM CI docker container.
# Inputs:
#   $1: Docker image
#   $2+: Command and arguments to run (relative to tflite-micro/ directory)
# Env:
#   TFLM_BOT_TOKEN: Required for docker login.
#   DOCKER_RUN_ENV: Optional additional flags for docker run (e.g. --env VAR=VAL)

set -e

if [ -z "$TFLM_BOT_TOKEN" ]; then
  echo "TFLM_BOT_TOKEN environment variable is not set."
  exit 1
fi

IMAGE=$1
shift

rm -rf .git
echo ${TFLM_BOT_TOKEN} | docker login ghcr.io -u tflm-bot --password-stdin

# Note: DOCKER_RUN_ENV is used without quotes to allow multiple flags.
docker run ${DOCKER_RUN_ENV} --rm -v `pwd`:/opt/tflite-micro ${IMAGE} \
  /bin/bash -c "cd /opt && tflite-micro/$*"
