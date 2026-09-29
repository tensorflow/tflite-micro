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
r"""Strips all nonessential strings from a TFLite file."""

import argparse

from tflite_micro.tensorflow.lite.tools import flatbuffer_utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input_tflite_file',
    required=True,
    help='Full path name to the input TFLite file.',
  )
  parser.add_argument(
    '--output_tflite_file',
    required=True,
    help='Full path name to the output stripped TFLite file.',
  )
  args, _ = parser.parse_known_args()

  model = flatbuffer_utils.read_model(args.input_tflite_file)
  flatbuffer_utils.strip_strings(model)
  flatbuffer_utils.write_model(model, args.output_tflite_file)


if __name__ == '__main__':
  main()
