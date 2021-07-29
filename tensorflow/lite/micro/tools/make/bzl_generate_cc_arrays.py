# Lint as: python2, python3
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
"""Generates a cc and header file from a bitmap or tflite file."""

import argparse

import generate_cc_arrays


def main():
    """Create cc sources with c arrays with data from each .tflite or .bmp."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_file', help='generated output file. This must be a .cc or a .h')
    parser.add_argument('input_file',
                        help='input bmp or tflite file to convert')
    args = parser.parse_args()
    size, cc_array = generate_cc_arrays.generate_array(args.input_file)
    generated_array_name = generate_cc_arrays.array_name(args.input_file)
    generate_cc_arrays.generate_file(args.output_file, generated_array_name,
                                     cc_array, size)


if __name__ == '__main__':
    main()
