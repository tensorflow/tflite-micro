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
"""Library for converting .tflite and .bmp files to cc arrays"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os


def generate_file(out_fname, array_name, out_string, size):
    ''' Write the out string containing an array of values to a variable in a cc
        file, and create a header file defining the same array. '''
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    if out_fname.endswith('.cc'):
        out_cc_file = open(out_fname, 'w')
        # Log cc file name for Make to include in the build.
        out_cc_file.write('#include "' + out_fname.replace('.cc', '.h') +
                          '"\n\n')
        out_cc_file.write('const unsigned char ' + array_name + '[] = {')
        out_cc_file.write(out_string)
        out_cc_file.write('};\n')
        out_cc_file.write('const unsigned int ' + array_name + '_size = ' +
                          str(size) + ';')
        out_cc_file.close()
    elif out_fname.endswith('.h'):
        out_hdr_file = open(out_fname, 'w')
        out_hdr_file.write('extern const unsigned char ' + array_name + '[];\n')
        out_hdr_file.write('extern const unsigned int ' + array_name + '_size' +
                           ';')
        out_hdr_file.close()
    else:
        raise ValueError('input file must be .tflite, .bmp')


def generate_array(input_fname):
    ''' Return array size and string containing an array of data from the input file. '''
    if input_fname.endswith('.tflite'):
        with open(input_fname, 'rb') as input_file:
            out_string = ''
            byte = input_file.read(1)
            size = 0
            while (byte != b''):
                out_string += '0x' + byte.hex() + ','
                byte = input_file.read(1)
                size += 1
            return [size, out_string]
    elif input_fname.endswith('.bmp'):
        img = Image.open(input_fname, mode='r')
        image_bytes = img.tobytes()
        out_string = ""
        for byte in image_bytes:
            out_string += hex(byte) + ","
        return [len(image_bytes), out_string]
    else:
        raise ValueError('input file must be .tflite, .bmp')


def array_name(input_fname):
    base_array_name = 'g_' + input_fname.split('.')[0].split('/')[-1]
    if input_fname.endswith('.tflite'):
        return base_array_name + '_model_data'
    elif input_fname.endswith('.bmp'):
        return base_array_name + '_image_data'
