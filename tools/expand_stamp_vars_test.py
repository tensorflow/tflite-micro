#!/usr/bin/env python3

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
# ----

# A test for the filter that expands Bazel workspace stamp variables.

from tflite_micro.tools import expand_stamp_vars

import io
import unittest


class FilterTest(unittest.TestCase):
  """A simple test of the expansion feature."""

  def test_basic(self):
    stamps = """
BUILD_STAMP_ONE value_one
BUILD_STAMP_TWO value_two
"""
    input = "This is {BUILD_STAMP_TWO}. This is {BUILD_STAMP_ONE}."
    golden = "This is value_two. This is value_one."

    istream = io.StringIO(input)
    ostream = io.StringIO()
    stamps = expand_stamp_vars.read_stamps(io.StringIO(stamps))
    expand_stamp_vars.expand(istream, ostream, stamps)

    self.assertEqual(ostream.getvalue(), golden)


if __name__ == "__main__":
  unittest.main()
