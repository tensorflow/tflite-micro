#!/usr/bin/env python3
#
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
#
"""Crosscheck the Micro interpreter against other interpreters.

Usage:
  crosscheck_tool --model <path>

Crosscheck the Micro interpreter's output against a reference interpreter's
output. The passed model and a randomly generated input tensor are fed both
interpreters, and the outputs are expected to be equality. See the crosscheck
module's documentation for more details.

Returns:
  0: if the outputs of Micro and the reference interpreter are equal.

  1: and prints the input, output, and difference to stderr if the
    outputs differ.
"""

from absl import app
from absl import flags
from absl import logging
import os
import sys

from tflite_micro.python.tflite_micro.test_utils import crosscheck

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, ".tflite model path")
flags.mark_flag_as_required("model")


def main(_):
  if not os.path.exists(FLAGS.model):
    print(os.getcwd())
    raise FileNotFoundError(FLAGS.model)

  result = crosscheck.versus_lite(tflite_path=FLAGS.model)

  if result:
    print("outputs match")
    return 0
  else:
    print("outputs differ")
    print(result, file=sys.stderr)
    return 1


if __name__ == "__main__":
  app.run(main)
