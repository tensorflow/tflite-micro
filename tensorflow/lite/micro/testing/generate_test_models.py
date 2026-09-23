# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python utility script to provide unit test model data."""

import os
import sys


def generate_conv_model(
    write_to_file=True, filename="/tmp/tf_micro_conv_test_model.int8.tflite"
):
  """Loads the pregenerated conv int8 model and optionally writes it to file."""
  model_path = os.path.join(
      os.path.dirname(__file__), "conv_test_model.tflite"
  )
  with open(model_path, "rb") as f:
    tflite_model = f.read()

  if write_to_file:
    with open(filename, "wb") as f:
      f.write(tflite_model)

  return tflite_model


def main(argv):
  del argv  # Unused for now
  generate_conv_model()


if __name__ == "__main__":
  main(sys.argv)
