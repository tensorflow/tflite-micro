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
"""Resolves non-system C/C++ includes to their full paths.

Used to generate ESP-IDF examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

import six

EXAMPLE_DIR_PATH = 'tensorflow/lite/micro/examples/'


def replace_esp_example_includes(line, source_path):
  """Updates any includes for local example files."""
  # Because the export process moves the example source and header files out of
  # their default locations into the top-level 'main' folder in the ESP-IDF
  # project, we have to update any include references to match.
  include_match = re.match(r'.*#include.*"(' + EXAMPLE_DIR_PATH + r'.*)"',
                           line)

  if include_match:
    # Compute the target path relative from the source's directory
    target_path = include_match.group(1)
    source_dirname = os.path.dirname(source_path)
    rel_to_target = os.path.relpath(target_path, start=source_dirname)

    line = '#include "%s"' % rel_to_target
  return line


def transform_esp_sources(input_lines, flags):
  """Transform sources for the ESP-IDF platform.

  Args:
    input_lines: A sequence of lines from the input file to process.
    flags: Flags indicating which transformation(s) to apply.

  Returns:
    The transformed output as a string.
  """
  output_lines = []
  for line in input_lines:
    if flags.is_example_source:
      line = replace_esp_example_includes(line, flags.source_path)
    output_lines.append(line)

  output_text = '\n'.join(output_lines)
  return output_text


def main(unused_args, flags):
  """Transforms the input source file to work when exported as example."""
  input_file_lines = sys.stdin.read().split('\n')

  output_text = ''
  if flags.platform == 'esp':
    output_text = transform_esp_sources(input_file_lines, flags)

  sys.stdout.write(output_text)


def parse_args():
  """Converts the raw arguments into accessible flags."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--platform',
                      choices=['esp'],
                      required=True,
                      help='Target platform.')
  parser.add_argument('--third_party_headers',
                      type=str,
                      default='',
                      help='Space-separated list of headers to resolve.')
  parser.add_argument('--is_example_ino',
                      dest='is_example_ino',
                      action='store_true',
                      help='Whether the destination is an example main ino.')
  parser.add_argument(
      '--is_example_source',
      dest='is_example_source',
      action='store_true',
      help='Whether the destination is an example cpp or header file.')
  parser.add_argument('--source_path',
                      type=str,
                      default='',
                      help='The relative path of the source code file.')
  flags, unparsed = parser.parse_known_args()

  main(unparsed, flags)


if __name__ == '__main__':
  parse_args()
