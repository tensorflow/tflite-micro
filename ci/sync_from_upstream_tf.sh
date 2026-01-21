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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/..
cd "${ROOT_DIR}"

if [[ ! -f "ci/tflite_files.txt" ]]; then
    echo "Error: ci/tflite_files.txt not found!"
    exit 1
fi

# Create a unique temp directory for the upstream clone to prevent collisions.
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT
echo "Syncing code using temp dir: ${TEMP_DIR}"

git clone https://github.com/google-ai-edge/LiteRT.git --depth=1 "${TEMP_DIR}/litert"

# separate standard code from converter code based on destination path.
SHARED_TFL_CODE=$(grep -v tensorflow/compiler/mlir/lite ci/tflite_files.txt)
SHARED_CONVERTER_CODE=$(grep tensorflow/compiler/mlir/lite ci/tflite_files.txt)

# Remove existing C/C++/Python files in target directories to prevent stale artifacts.
# Preserves 'experimental' and 'micro' directories.
echo "Cleaning old files..."
find tensorflow/lite/ -type d \( -path tensorflow/lite/experimental -o -path tensorflow/lite/micro \) -prune -false \
    -o -name "*.cc" -o -name "*.c" -o -name "*.h" -o -name "*.py" -o -name "*.fbs" \
    -exec rm -f {} +

# Helper wrapper for sed to handle differences between GNU (Linux) and BSD (macOS).
run_sed() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Copies files from upstream and patches include paths.
# Handles the reverse mapping of Local Destination Path -> Upstream Source Path.
process_files() {
    local file_list=$1
    local is_converter=$2

    for local_filepath in ${file_list}; do
        local upstream_filepath=""

        # Calculate the upstream path by reversing the local directory structure.
        if [ "$is_converter" = true ]; then
             upstream_filepath=${local_filepath//tensorflow\/compiler\/mlir\/lite\//tflite\/converter\/}
        else
             upstream_filepath=${local_filepath//tensorflow\/lite\//tflite\/}
        fi

        if [ ! -f "${TEMP_DIR}/litert/${upstream_filepath}" ]; then
            echo "WARNING: Upstream file not found: ${upstream_filepath} (requested by ${local_filepath})"
            continue
        fi

        mkdir -p "$(dirname "${local_filepath}")"
        /bin/cp "${TEMP_DIR}/litert/${upstream_filepath}" "${local_filepath}"

        # Patch include paths within the source code.
        run_sed 's/tflite\/converter\//tensorflow\/compiler\/mlir\/lite\//' "${local_filepath}"
        run_sed 's/tflite\//tensorflow\/lite\//' "${local_filepath}"
    done
}

echo "Copying Standard TFLite Code..."
process_files "${SHARED_TFL_CODE}" false

echo "Copying Converter Code..."
process_files "${SHARED_CONVERTER_CODE}" true

echo "Reverting divergent files..."
# Revert specific files where tflite-micro maintains local modifications.
# https://github.com/tensorflow/tflite-micro/pull/8
git checkout tensorflow/lite/kernels/internal/optimized/neon_check.h
# http://b/149862813
git checkout tensorflow/lite/kernels/internal/runtime_shape.h
git checkout tensorflow/lite/kernels/internal/runtime_shape.cc
# http://b/187728891
git checkout tensorflow/lite/kernels/op_macros.h
# http://b/242077843
git checkout tensorflow/lite/kernels/internal/tensor_utils.cc

echo "Building schemas..."
# Generate Flatbuffer schemas (Python and C++) and copy them to the source tree
# so they work with Makefiles.
bazel build tensorflow/lite/python:schema_py
/bin/cp bazel-bin/tensorflow/lite/python/schema_py_generated.py tensorflow/lite/python/schema_py_generated.py

bazel build tensorflow/compiler/mlir/lite/schema:schema_fbs_srcs
/bin/cp ./bazel-bin/tensorflow/compiler/mlir/lite/schema/schema_generated.h tensorflow/lite/schema/schema_generated.h

bazel clean

echo "Patching Python imports..."
# The shared Python code requires modified imports to work within the tflite-micro namespace.
find tensorflow/lite/tools tensorflow/lite/python -name "*.py" -print0 | while IFS= read -r -d '' py_file; do
    run_sed 's/from tflite/from tflite_micro\.tensorflow\.lite/' "${py_file}"
done

# The microfrontend is maintained as a fork, so we revert to the local version.
git checkout tensorflow/lite/experimental/

echo "Sync complete."
