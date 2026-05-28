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
"""Single source of truth for mapping a TFLite TensorType to a numpy dtype.

Compression tooling reads tensor buffer bytes as numpy arrays, so it needs to
know the element type. Only the TensorTypes with a clean numpy equivalent are
mapped; anything else raises rather than silently guessing a type.
"""

import numpy as np

from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

# TFLite buffers are little-endian, so the dtypes are pinned to little-endian
# byte order to keep np.frombuffer correct on any host.
_TO_NUMPY = {
    tflite.TensorType.FLOAT16: np.dtype("<f2"),
    tflite.TensorType.FLOAT32: np.dtype("<f4"),
    tflite.TensorType.FLOAT64: np.dtype("<f8"),
    tflite.TensorType.INT8: np.dtype("<i1"),
    tflite.TensorType.INT16: np.dtype("<i2"),
    tflite.TensorType.INT32: np.dtype("<i4"),
    tflite.TensorType.INT64: np.dtype("<i8"),
    tflite.TensorType.UINT8: np.dtype("<u1"),
    tflite.TensorType.UINT16: np.dtype("<u2"),
    tflite.TensorType.UINT32: np.dtype("<u4"),
    tflite.TensorType.UINT64: np.dtype("<u8"),
}

# TensorType value -> name, for readable error messages.
_NAMES = {
    value: name
    for name, value in vars(tflite.TensorType).items()
    if not name.startswith("_")
}


def to_numpy(tensor_type: int) -> np.dtype:
  """Return the little-endian numpy dtype for a TFLite TensorType.

  Raises:
    ValueError: if the type has no clean numpy equivalent (e.g. STRING,
        RESOURCE, VARIANT, BFLOAT16, or the sub-byte INT4/UINT4/INT2 types).
  """
  try:
    return _TO_NUMPY[tensor_type]
  except KeyError:
    name = _NAMES.get(tensor_type, "?")
    raise ValueError(
        f"no numpy dtype for TFLite TensorType {name} ({tensor_type})")
