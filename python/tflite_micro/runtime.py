# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Python package for TFLM Python Interpreter"""

import enum
import os
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils
from tflite_micro.python.tflite_micro import _runtime


class InterpreterConfig(enum.Enum):
  """There are two mutually exclusive types of way you could use the TFLM python

  interpreter, this enum is made so that users can clearly choose between the
  two
  different usage method for the interpreter.

  The first default way is kRecordingAllocation where all memory usage by the
  interpreter is recorded on inference. When using this config the GetTensor()
  api is disabled by the interpreter since this interpreter configuration
  doesn’t
  guarantee that the valid data for all tensors is available post inference.

  The second way is kPreserveAllTensors where the GetTensor() api is disabled by
  the interpreter since this interpreter configuration doesn’t guarantee that
  the
  valid data for all tensors is available post inference. But the memory usage
  by
  the interpreter won’t be recorded on inference.

  Usage:

  default_interpreter = Interpreter(…
        intrepreter_config=InterpreterConfig.kAllocationRecording)

  preserve_interpreter = Interpreter(…
        intrepreter_config=InterpreterConfig.kPreserveAllTensors)
  """

  kAllocationRecording = 0
  kPreserveAllTensors = 1


#TODO(b/297118768): Once Korko Docker contrainer for ubuntu x86 has imutabledict
# added to it, this should be turned into an immutabledict.
_ENUM_TRANSLATOR = {
    InterpreterConfig.kAllocationRecording:
    (_runtime.PythonInterpreterConfig.kAllocationRecording),
    InterpreterConfig.kPreserveAllTensors:
    (_runtime.PythonInterpreterConfig.kPreserveAllTensors),
}


class Interpreter(object):

  def __init__(
      self,
      model_data,
      custom_op_registerers,
      arena_size,
      intrepreter_config=InterpreterConfig.kAllocationRecording,
  ):
    if model_data is None:
      raise ValueError("Model must not be None")

    if not isinstance(custom_op_registerers, list) or not all(
        isinstance(s, str) for s in custom_op_registerers):
      raise ValueError("Custom ops registerers must be a list of strings")

    # This is a heuristic to ensure that the arena is sufficiently sized.
    if arena_size is None:
      arena_size = len(model_data) * 10
    # Some models make use of resource variables ops, get the count here
    num_resource_variables = flatbuffer_utils.count_resource_variables(
        model_data)
    print("Number of resource variables the model uses = ",
          num_resource_variables)

    self._interpreter = _runtime.InterpreterWrapper(
        model_data,
        custom_op_registerers,
        arena_size,
        num_resource_variables,
        _ENUM_TRANSLATOR[intrepreter_config],
    )

  @classmethod
  def from_file(
      self,
      model_path,
      custom_op_registerers=[],
      arena_size=None,
      intrepreter_config=InterpreterConfig.kAllocationRecording,
  ):
    """Instantiates a TFLM interpreter from a model .tflite filepath.

    Args:
      model_path: Filepath to the .tflite model
      custom_op_registerers: List of strings, each of which is the name of a
        custom OP registerer
      arena_size: Tensor arena size in bytes. If unused, tensor arena size will
        default to 10 times the model size.

    Returns:
      An Interpreter instance
    """
    if model_path is None or not os.path.isfile(model_path):
      raise ValueError("Invalid model file path")

    with open(model_path, "rb") as f:
      model_data = f.read()

    return Interpreter(
        model_data,
        custom_op_registerers,
        arena_size,
        intrepreter_config,
    )

  @classmethod
  def from_bytes(
      self,
      model_data,
      custom_op_registerers=[],
      arena_size=None,
      intrepreter_config=InterpreterConfig.kAllocationRecording,
  ):
    """Instantiates a TFLM interpreter from a model in byte array.

    Args:
      model_data: Model in byte array format
      custom_op_registerers: List of strings, each of which is the name of a
        custom OP registerer
      arena_size: Tensor arena size in bytes. If unused, tensor arena size will
        default to 10 times the model size.

    Returns:
      An Interpreter instance
    """

    return Interpreter(
        model_data,
        custom_op_registerers,
        arena_size,
        intrepreter_config,
    )

  def print_allocations(self):
    """Invoke the RecordingMicroAllocator to print the arena usage.

    This should be called after `invoke()`.

    Returns:
      This method does not return anything, but It dumps the arena
      usage to stderr.
    """
    self._interpreter.PrintAllocations()

  def invoke(self):
    """Invoke the TFLM interpreter to run an inference.

    This should be called after `set_input()`.

    Returns:
      Status code of the C++ invoke function. A RuntimeError will be raised as
      well upon any error.
    """
    return self._interpreter.Invoke()

  def reset(self):
    """Reset the model state to be what you would expect when the interpreter is first

    created. i.e. after Init and Prepare is called for the very first time.

    This should be called after invoke stateful model like LSTM.

    Returns:
      Status code of the C++ invoke function. A RuntimeError will be raised as
      well upon any error.
    """
    return self._interpreter.Reset()

  def set_input(self, input_data, index):
    """Set input data into input tensor.

    This should be called before `invoke()`.

    Args:
      input_data: Input data in numpy array format. The numpy array format is
        chosen to be consistent with TFLite interpreter.
      index: An integer between 0 and the number of input tensors (exclusive)
        consistent with the order defined in the list of inputs in the .tflite
        model
    """
    if input_data is None:
      raise ValueError("Input data must not be None")
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    self._interpreter.SetInputTensor(input_data, index)

  def get_output(self, index):
    """Get data from output tensor.

    The output data correspond to the most recent `invoke()`.

    Args:
      index: An integer between 0 and the number of output tensors (exclusive)
        consistent with the order defined in the list of outputs in the .tflite
        model

    Returns:
      Output data in numpy array format. The numpy array format is chosen to
      be consistent with TFLite interpreter.
    """
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    return self._interpreter.GetOutputTensor(index)

  def GetTensor(self, tensor_index, subgraph_index):
    return self._interpreter.GetTensor(tensor_index, subgraph_index)

  def get_input_details(self, index):
    """Get input tensor information

    Args:
        index (int): An integer between 0 and the number of output tensors
          (exclusive) consistent with the order defined in the list of outputs
          in the .tflite model

    Returns:
        A dictionary from input index to tensor details where each item is a
        dictionary with details about an input tensor. Each dictionary contains
        the following fields that describe the tensor:
        + `shape`: The shape of the tensor.
        + `dtype`: The numpy data type (such as `np.int32` or `np.uint8`).
        + `quantization_parameters`: A dictionary of parameters used to quantize
          the tensor:
          ~ `scales`: List of scales (one if per-tensor quantization).
          ~ `zero_points`: List of zero_points (one if per-tensor quantization).
          ~ `quantized_dimension`: Specifies the dimension of per-axis
          quantization, in the case of multiple scales/zero_points.

    """
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    return self._interpreter.GetInputTensorDetails(index)

  def get_output_details(self, index):
    """Get output tensor information

    Args:
        index (int): An integer between 0 and the number of output tensors
          (exclusive) consistent with the order defined in the list of outputs
          in the .tflite model

    Returns:
        A dictionary from input index to tensor details where each item is a
        dictionary with details about an input tensor. Each dictionary contains
        the following fields that describe the tensor:
        + `shape`: The shape of the tensor.
        + `dtype`: The numpy data type (such as `np.int32` or `np.uint8`).
        + `quantization_parameters`: A dictionary of parameters used to quantize
          the tensor:
          ~ `scales`: List of scales (one if per-tensor quantization).
          ~ `zero_points`: List of zero_points (one if per-tensor quantization).
          ~ `quantized_dimension`: Specifies the dimension of per-axis
          quantization, in the case of multiple scales/zero_points.

    """
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    return self._interpreter.GetOutputTensorDetails(index)
