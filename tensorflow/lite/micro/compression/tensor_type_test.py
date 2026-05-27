# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

import unittest

import numpy as np

from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite
from tflite_micro.tensorflow.lite.micro.compression import tensor_type


class TensorTypeTest(unittest.TestCase):

  def test_maps_known_types_to_little_endian_dtypes(self):
    self.assertEqual(tensor_type.to_numpy(tflite.TensorType.INT8),
                     np.dtype("<i1"))
    self.assertEqual(tensor_type.to_numpy(tflite.TensorType.UINT32),
                     np.dtype("<u4"))
    self.assertEqual(tensor_type.to_numpy(tflite.TensorType.FLOAT32),
                     np.dtype("<f4"))

  def test_dtype_itemsize_matches_type_width(self):
    # Reading buffers depends on the dtype having the right element size.
    self.assertEqual(tensor_type.to_numpy(tflite.TensorType.INT16).itemsize, 2)
    self.assertEqual(
        tensor_type.to_numpy(tflite.TensorType.FLOAT64).itemsize, 8)

  def test_raises_on_type_without_numpy_equivalent(self):
    with self.assertRaises(ValueError):
      tensor_type.to_numpy(tflite.TensorType.STRING)


if __name__ == "__main__":
  unittest.main()
