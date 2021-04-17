#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#
# Sync's the shared TfLite / TFLM code from the upstream Tensorflow repo.
#
# While the standalone TFLM repo is under development, we are also sync'ing all
# of the TFLM code via this script.
#

set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/..
cd "${ROOT_DIR}"

rm -rf /tmp/tensorflow
rm -rf /tmp/tflm-tree

git clone https://github.com/tensorflow/tensorflow.git --depth=1 /tmp/tensorflow

cd /tmp/tensorflow/
# TODO(b/184886633): the downloads should happen as part of the create_tflm_tree
# script.
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py /tmp/tflm-tree
cd -

rsync -r --delete /tmp/tflm-tree/tensorflow/lite tensorflow/

# The entire micro directory will be copied from upstream TF (instead of being
# copied from the output of the project generation).
git checkout tensorflow/lite/micro/

# TfLite BUILD files are manually maintained in the TFLM repo.
git checkout \
  tensorflow/BUILD \
  tensorflow/lite/BUILD \
  tensorflow/lite/build_def.bzl \
  tensorflow/lite/c/BUILD \
  tensorflow/lite/core/api/BUILD \
  tensorflow/lite/kernels/BUILD \
  tensorflow/lite/kernels/internal/BUILD \
  tensorflow/lite/schema/BUILD

rsync -r --delete /tmp/tensorflow/tensorflow/lite/micro tensorflow/lite/

# TODO(b/184876027): properly handle the micro_speech example and its
# dependencies.
rm -rf tensorflow/lite/micro/examples/micro_speech

# TODO(b/184884735): enable the person_detection example. We need some of the
# sources from the person_detection example for the benchmarks and so are
# currently only removing the Makefile and the BUILD files.
rm -rf tensorflow/lite/micro/examples/person_detection/Makefile
rm -rf tensorflow/lite/micro/examples/person_detection/BUILD
rm -rf tensorflow/lite/micro/examples/person_detection/utils/BUILD

rm -rf tensorflow/lite/micro/tools/ci_build/tflm_bazel

# Any TFLM-repo specific files that are not in upstream TF will be deleted with
# the rsync command and any files whose source of truth is the new TFLM repo
# should be manually restored.
git checkout \
  tensorflow/lite/micro/tools/ci_build/test_all.sh \
  tensorflow/lite/micro/tools/ci_build/test_bazel.sh

