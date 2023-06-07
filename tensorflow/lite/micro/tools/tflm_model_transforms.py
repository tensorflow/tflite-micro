# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Runs TFLM specific transformations to reduce model size on a .tflite model."""

from absl import app
from absl import flags
from absl import logging

from tflite_micro.tensorflow.lite.micro.tools import tflm_model_transforms_lib

# Usage information:
# Default:
#   `bazel run tensorflow/lite/micro/tools:tflm_model_transforms -- \
#     --input_model_path=</path/to/my_model.tflite>`
# output will be located at: /path/to/my_model_tflm_optimized.tflite

_INPUT_MODEL_PATH = flags.DEFINE_string(
    "input_model_path",
    None,
    ".tflite input model path",
    required=True,
)

_SAVE_INTERMEDIATE_MODELS = flags.DEFINE_bool(
    "save_intermediate_models",
    False,
    "optional config to save models between different transforms. Models are"
    " saved to a /tmp/ directory and tested at each stage.",
)

_TEST_TRANSFORMED_MODELS = flags.DEFINE_bool(
    "test_transformed_model",
    True,
    "optional config to enable/disable testing models on random data and"
    " asserting equivalent output.",
)

_OUTPUT_MODEL_PATH = flags.DEFINE_string(
    "output_model_path",
    None,
    ".tflite output path. Leave blank if same as input+_tflm_optimized.tflite",
)


def main(_) -> None:
  output_model_path = _OUTPUT_MODEL_PATH.value or (
      _INPUT_MODEL_PATH.value.split(".tflite")[0] + "_tflm_optimized.tflite")

  logging.info("\n--Running TFLM optimizations on: %s",
               _INPUT_MODEL_PATH.value)
  tflm_model_transforms_lib.run_all_transformations(
      _INPUT_MODEL_PATH.value,
      output_model_path,
      _SAVE_INTERMEDIATE_MODELS.value,
      _TEST_TRANSFORMED_MODELS.value,
  )


if __name__ == "__main__":
  app.run(main)
