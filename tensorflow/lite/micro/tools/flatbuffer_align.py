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
# ==============================================================================
"""Helper module to align tflite flatbuffers using C-API in libtflm.so."""

import ctypes
from tflite_micro.python.tflite_micro.runtime import _lib

_lib.TFLM_AlignFlatbuffer.restype = ctypes.c_int
_lib.TFLM_AlignFlatbuffer.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
]


def align_tflite_model(input_model_path: str, output_model_path: str):
  """Aligns the tflite flatbuffer to 16 bytes."""
  err_buf = ctypes.create_string_buffer(512)
  status = _lib.TFLM_AlignFlatbuffer(
      input_model_path.encode("utf-8"),
      output_model_path.encode("utf-8"),
      err_buf,
      len(err_buf),
  )
  if status != 0:
    raise RuntimeError(err_buf.value.decode("utf-8") or "Failed to align model.")
