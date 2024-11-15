#!/bin/sh

# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

# Run this script outside of bazel to regenerate the header-only C++ library
# used for reading the metadata flatbuffer and copy it to the source tree as
# $saved. See the bazel target $label.

set -e

workspace=$(bazel info workspace)
package=tensorflow/lite/micro/compression
label=//$package:metadata_cc
generated=$workspace/bazel-bin/$package/metadata_generated.h
saved=$workspace/$package/metadata_saved.h

bazel build $label
cp $generated $saved
chmod 664 $saved
