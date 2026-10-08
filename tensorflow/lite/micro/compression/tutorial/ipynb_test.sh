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

# Ensure the consistency between the tutorial's markdown source and the
# notebook generated from it that has been saved to git. Regenerate the
# notebook and compare it to the saved version. Fail the test (return 1) if
# they are not identical. See the bazel target
# ":ipynb".

set -e

saved=$1
generated=$2

if diff -q $saved $generated
then
    exit 0
else
    cat <<HERE
    FAILURE: "$(basename $saved)" in the source tree does not match the
    notebook generated from the current "mnist_compression_tutorial.md". Run

        bazel run //tensorflow/lite/micro/compression/tutorial:ipynb.update

    to update "$(basename $saved)", and commit the changes to git. See the
    target ":ipynb" for more detail.
HERE
    exit 1
fi
