#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Helper to run a script inside a TFLM CI docker container.
#
# Inputs:
#   $1: Docker image
#   $2+: Command and arguments to run (relative to tflite-micro/ directory)
# Env:
#   TFLM_BOT_TOKEN: Required for docker login.
#   DOCKER_RUN_ENV: Optional additional flags for docker run (e.g. --env VAR=VAL)

set -e
set -u

if [ -z "${TFLM_BOT_TOKEN:-}" ]; then
  echo "TFLM_BOT_TOKEN environment variable is not set."
  exit 1
fi

IMAGE=$1
shift

echo "${TFLM_BOT_TOKEN}" | docker login ghcr.io -u tflm-bot --password-stdin

# Note: DOCKER_RUN_ENV is used without quotes to allow multiple flags.
docker run ${DOCKER_RUN_ENV:-} --rm -v "$(pwd)":/opt/tflite-micro "${IMAGE}" \
  /bin/bash -c "cd /opt && tflite-micro/$*"
