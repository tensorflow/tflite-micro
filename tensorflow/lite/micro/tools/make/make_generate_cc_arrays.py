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
"""Generates cc and header files from bitmap or tflite files."""

import argparse
import os

import generate_cc_arrays


def main():
    """Create cc sources with c arrays with data from each .tflite or .bmp."""
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='base directory for all outputs')
    parser.add_argument('input_files',
                        nargs='+',
                        help='input bmp or tflite files to convert')
    args = parser.parse_args()
    for input_file in args.input_files:
        output_base_fname = os.path.join(args.output_dir,
                                         input_file.split('.')[0])
        if input_file.endswith('.tflite'):
            output_base_fname = output_base_fname + '_model_data'
        elif input_file.endswith('.bmp'):
            output_base_fname = output_base_fname + '_image_data'
        else:
            raise ValueError('input file must be .tflite, .bmp')

        output_cc_fname = output_base_fname + '.cc'
        # Print output cc filename for Make to include it in the build.
        print(output_cc_fname)
        output_hdr_fname = output_base_fname + '.h'
        size, cc_array = generate_cc_arrays.generate_array(input_file)
        generated_array_name = generate_cc_arrays.array_name(input_file)
        generate_cc_arrays.generate_file(output_cc_fname, generated_array_name,
                                         cc_array, size)
        generate_cc_arrays.generate_file(output_hdr_fname, generated_array_name,
                                         cc_array, size)


if __name__ == '__main__':
    main()
