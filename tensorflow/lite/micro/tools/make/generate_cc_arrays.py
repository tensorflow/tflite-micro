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
"""Converts .csv, .tflite, .wav and .bmp files to cc arrays"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import argparse
import io
import os


def generate_cc_file(out_fname, array_name, out_string, size):
    ''' Write the out string containing an array of values to a variable in a cc
        file, and create a header file defining the same array. '''
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    out_cc_file = open(out_fname + '.cc', 'w')
    print(
        out_cc_file.name)  # Log cc file name for Make to include in the build.
    out_cc_file.write('#include "' + out_fname + '.h"\n\n')
    out_cc_file.write('const unsigned char ' + array_name + '[] = {')
    out_cc_file.write(out_string)
    out_cc_file.write('};\n')
    out_cc_file.write('const unsigned int ' + array_name + '_size = ' +
                      str(size) + ';')
    out_cc_file.close()


def generate_hdr_file(out_fname, array_name):
    ''' Write the out string containing an array of values to a variable in a cc
        file, and create a header file defining the same array. '''
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    out_hdr_file = open(out_fname + '.h', 'w')
    out_hdr_file.write('extern const unsigned char ' + array_name + '[];\n')
    out_hdr_file.write('extern const unsigned int ' + array_name + '_size' +
                       ';')
    out_hdr_file.close()


def tflite_to_cc(tflite_fname, out_fname, generate_cc, generate_hdr):
    ''' Serialize a tflite file to a header and cc file for use with TFLM. '''
    with open(tflite_fname, 'rb') as tflite_file:
        out_string = ''
        byte = tflite_file.read(1)
        size = 0
        while (byte != b''):
            out_string += '0x' + byte.hex() + ','
            byte = tflite_file.read(1)
            size += 1
        # Array name is 'g_<model name without path or extension>_model_data'
        array_name = 'g_' + tflite_fname.split('.')[0].split(
            '/')[-1] + '_model_data'
        out_fname = out_fname + '_model_data'
        if generate_hdr:
            generate_hdr_file(out_fname, array_name)
        if generate_cc:
            generate_cc_file(out_fname, array_name, out_string, size)


def bmp_to_cc(bmp_fname, out_fname, generate_cc, generate_hdr):
    ''' Serialize an image to a header and cc file for use with TFLM.'''
    img = Image.open(bmp_fname, mode='r')
    image_bytes = img.tobytes()
    out_string = ""
    for byte in image_bytes:
        out_string += hex(byte) + ","
    # Array name is 'g_<file name without path or extension>_image_data'
    array_name = 'g_' + bmp_fname.split('.')[0].split('/')[-1] + '_image_data'
    out_fname = out_fname + '_image_data'
    if generate_hdr:
        generate_hdr_file(out_fname, array_name)
    if generate_cc:
        generate_cc_file(out_fname, array_name, out_string, len(image_bytes))


def generate_cc_and_hdrs(generator_output,
                         input_files,
                         cc_only=False,
                         hdr_only=False):
    generate_cc = True
    generate_hdr = True
    if hdr_only:
        generate_cc = False
    if cc_only:
        generate_hdr = False

    for input_file in input_files:
        if os.path.isdir(generator_output):
          output_dir = os.path.join(generator_output,
                                  input_file.split('.')[0])
        else:
          output_dir = generator_output
        if input_file.endswith('.tflite'):
            tflite_to_cc(input_file, output_dir, generate_cc, generate_hdr)
        elif input_file.endswith('.bmp'):
            bmp_to_cc(input_file, output_dir, generate_cc, generate_hdr)
        else:
            raise ValueError('input file must be .tflite, .bmp')


def main():
    '''Create cc sources with c arrays with data from each .tflite or .bmp.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'generator_output_dir',
        help=
        'location where generated files are saved. Generated files will be saved to <output_dir>/<input file without file extension>.cc/.h'
    )
    parser.add_argument('inputs',
                        nargs='+',
                        help='input bmp or tflite files to convert')
    parser.add_argument('--hdr_only', help='generate only a single header file')
    parser.add_argument('--cc_only', help='generate only a single cc file')
    args = parser.parse_args()
    generate_cc_and_hdrs(args.generator_output_dir, args.inputs, args.cc_only,
                         args.hdr_only)


if __name__ == '__main__':
    main()
