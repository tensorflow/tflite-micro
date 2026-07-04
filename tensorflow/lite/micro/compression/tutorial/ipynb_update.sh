#!/bin/sh

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

# Copy the notebook generated from the tutorial's markdown source into the
# source tree as the saved notebook. Build actions cannot write to the source
# tree, but executables started by `bazel run` can, so run this via the bazel
# target ":ipynb.update":
#
#   bazel run //tensorflow/lite/micro/compression/tutorial:ipynb.update
#
# See the bazel target ":ipynb".

set -e

generated=$1
saved=$BUILD_WORKSPACE_DIRECTORY/tensorflow/lite/micro/compression/tutorial/mnist_compression_tutorial.ipynb

cp $generated $saved
chmod 664 $saved
