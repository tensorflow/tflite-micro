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
""" Utility functions and classes for code generation. """

from typing import Any, Generator, Iterable, List, Optional, Sequence, Tuple
import string
import textwrap
import itertools


def to_pascal_case(s: str) -> str:
  """ Basic function for converting snake_case to PascalCase. """
  # This isn't perfect, as there might be some cases where we want underscores
  # to remain if they are used as number separators.
  return s.title().replace('_', '')


def bool_to_c_str(b: bool) -> str:
  """ Convert a python bool value to a C bool string. Ie, False -> 'false' """
  return str(b).lower()


def split_into_chunks(
    data: Iterable[Any],
    chunk_size: int) -> Generator[Tuple[Any, ...], None, None]:
  """Splits an iterable into chunks of a given size."""
  data_iterator = iter(data)
  while True:
    chunk = tuple(itertools.islice(data_iterator, chunk_size))
    if not chunk:
      break
    yield chunk


def generate_c_int_array(indent: str, int_type: str, name: str,
                         ints: Sequence[int]) -> str:
  int_strs = ['{}'.format(i) for i in ints]

  # Try to do it on a single line first
  single_line_array_template = string.Template(
      "constexpr ${int_type} ${name}[${size}] = {${data}};")
  single_line = textwrap.indent(
      single_line_array_template.substitute(int_type=int_type,
                                            name=name,
                                            size=len(int_strs),
                                            data=', '.join(int_strs)), indent)

  if len(single_line) < 81:
    return single_line

  # Couldn't fit, so split it across multiple lines
  multi_line_array_template = string.Template(
      "constexpr ${int_type} ${name}[${size}] = {\n"
      "${body}\n"
      "};\n")

  lines = []
  for int_strs_for_line in split_into_chunks(int_strs, 12):
    ints_segment = ', '.join(int_strs_for_line)
    lines.append(f'    {ints_segment},')

  return textwrap.indent(
      multi_line_array_template.substitute(int_type=int_type,
                                           name=name,
                                           size=len(ints),
                                           body='\n'.join(lines)), indent)


class IntArray(object):
  """ A helper class for generating int arrays that can be used to provide the
      backing storage for a TfLiteIntArray. """

  def __init__(self, data: List[int]):
    self._data = data

  def generate_c_struct(self, type_name: str,
                        variable_name: Optional[str]) -> str:
    struct_template = string.Template("struct ${type_name} {\n"
                                      "  int size = ${size};\n"
                                      "  int data[${size}] = {${data}};\n"
                                      "}")
    # TODO(rjascani): Make this pretty print in multi-line chunks
    int_strs = ['{}'.format(i) for i in self._data]
    c_struct_str = struct_template.substitute(type_name=type_name,
                                              size=len(int_strs),
                                              data=', '.join(int_strs))
    if variable_name:
      return c_struct_str + " {};".format(variable_name)
    return c_struct_str + ";"
