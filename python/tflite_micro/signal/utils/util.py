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
"""Python utility functions."""

import tensorflow as tf
from tflite_micro.python.tflite_micro import runtime


# TODO(b/286889497): find better name and place for this function.
def get_tflm_interpreter(concrete_function, trackable_obj):
  """Initialize a TFLite interpreter with a concrete function.

  Args:
    concrete_function: A concrete function

  Returns:
    TFLite interpreter object
  """
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_function], trackable_obj
  )
  converter.allow_custom_ops = True
  tflite_model = converter.convert()

  return runtime.Interpreter.from_bytes(tflite_model, arena_size=500000)


def load_custom_op(name):
  try:
    from tflite_micro.python.tflite_micro.signal import gen_delay_op
    from tflite_micro.python.tflite_micro.signal import gen_energy_op
    from tflite_micro.python.tflite_micro.signal import gen_fft_ops
    from tflite_micro.python.tflite_micro.signal import gen_filter_bank_ops
    from tflite_micro.python.tflite_micro.signal import gen_framer_op
    from tflite_micro.python.tflite_micro.signal import gen_overlap_add_op
    from tflite_micro.python.tflite_micro.signal import gen_pcan_op
    from tflite_micro.python.tflite_micro.signal import gen_stacker_op
    from tflite_micro.python.tflite_micro.signal import gen_window_op

    return {
      'delay_op.so': gen_delay_op,
      'energy_op.so': gen_energy_op,
      'fft_ops.so': gen_fft_ops,
      'filter_bank_ops.so': gen_filter_bank_ops,
      'framer_op.so': gen_framer_op,
      'overlap_add_op.so': gen_overlap_add_op,
      'pcan_op.so': gen_pcan_op,
      'stacker_op.so': gen_stacker_op,
      'window_op.so': gen_window_op,
    }[name]
  except ImportError:
    from tensorflow.python.framework import load_library
    from tensorflow.python.platform import resource_loader

    return load_library.load_op_library(
      resource_loader.get_path_to_datafile('../ops/_' + name)
    )
