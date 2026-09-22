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

import argparse
import logging

from tflite_micro.tensorflow.lite.micro.tools import tflm_model_transforms_lib

# Usage information:
# Default:
#   `bazel run tensorflow/lite/micro/tools:tflm_model_transforms -- \
#     --input_model_path=</path/to/my_model.tflite>`
# output will be located at: /path/to/my_model_tflm_optimized.tflite


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input_model_path",
    required=True,
    help=".tflite input model path",
  )
  parser.add_argument(
    "--save_intermediate_models",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "optional config to save models between different transforms. Models are"
      " saved to a /tmp/ directory and tested at each stage."
    ),
  )
  parser.add_argument(
    "--test_transformed_model",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "optional config to enable/disable testing models on random data and"
      " asserting equivalent output."
    ),
  )
  parser.add_argument(
    "--output_model_path",
    default=None,
    help=(
      ".tflite output path. Leave blank if same as input+_tflm_optimized.tflite"
    ),
  )
  args, _ = parser.parse_known_args()

  output_model_path = args.output_model_path or (
    args.input_model_path.split(".tflite")[0] + "_tflm_optimized.tflite"
  )

  logging.info("\n--Running TFLM optimizations on: %s", args.input_model_path)
  tflm_model_transforms_lib.run_all_transformations(
    args.input_model_path,
    output_model_path,
    args.save_intermediate_models,
    args.test_transformed_model,
  )


if __name__ == "__main__":
  main()
