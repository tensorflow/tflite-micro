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

import os
import numpy as np
import tensorflow as tf
from tflite_micro.python.tflite_micro import runtime


class _TfLiteInterpreterWrapper:
  """Wraps TFLite InterpreterWithCustomOps with set_input/get_output helpers."""

  def __init__(self, interpreter):
    self._interpreter = interpreter
    self._input_details = interpreter.get_input_details()
    self._output_details = interpreter.get_output_details()

  def set_input(self, input_data, index):
    self._interpreter.set_tensor(
      self._input_details[index]['index'], np.asarray(input_data)
    )

  def invoke(self):
    self._interpreter.invoke()

  def get_output(self, index):
    return self._interpreter.get_tensor(self._output_details[index]['index'])

  def __getattr__(self, name):
    return getattr(self._interpreter, name)


# TODO(b/286889497): find better name and place for this function.
def get_tflm_interpreter(concrete_function, trackable_obj):
  """Initialize a TFLM (or TFLite if TFLM_SIGNAL_USE_TFLITE=1) interpreter.

  Args:
    concrete_function: A concrete function
    trackable_obj: Trackable object passed to TFLiteConverter.

  Returns:
    Interpreter object
  """
  if os.environ.get('TFLM_SIGNAL_USE_TFLITE') == '1':
    return get_tflite_interpreter(concrete_function, trackable_obj)

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_function], trackable_obj
  )
  converter.allow_custom_ops = True
  tflite_model = converter.convert()

  return runtime.Interpreter.from_bytes(tflite_model, arena_size=500000)


def get_tflite_interpreter(
  concrete_function, trackable_obj, custom_op_registerers=None
):
  """Initialize a TFLite interpreter with SignalOpsRegisterer.

  Args:
    concrete_function: A concrete function
    trackable_obj: Trackable object passed to TFLiteConverter.
    custom_op_registerers: Additional custom op registerer names or functions.

  Returns:
    Wrapped TFLite InterpreterWithCustomOps object.
  """
  import ai_edge_litert.interpreter as tflite_interp
  from tflite_micro.signal.lite.register import pywrap_signal_ops

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_function], trackable_obj
  )
  converter.allow_custom_ops = True
  tflite_model = converter.convert()

  registerers = list(custom_op_registerers or [])
  if (
    'SignalOpsRegisterer' not in registerers
    and pywrap_signal_ops.SignalOpsRegisterer not in registerers
  ):
    registerers.append(pywrap_signal_ops.SignalOpsRegisterer)
  interpreter = tflite_interp.InterpreterWithCustomOps(
    model_content=tflite_model,
    custom_op_registerers=registerers,
  )
  interpreter.allocate_tensors()
  return _TfLiteInterpreterWrapper(interpreter)


def load_custom_op(name):
  try:
    import importlib
    from tflite_micro.python.tflite_micro import signal

    return importlib.import_module(
      f'{signal.__name__}.gen_{name.removesuffix(".so")}'
    )
  except ImportError:
    return tf.load_op_library(
      os.path.join(os.path.dirname(__file__), '..', 'ops', '_' + name)
    )
