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
"""Runs TFLM specific transformations to reduce model size on a .tflite model."""

import sys
import unittest

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from tflite_micro.tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.platform import gfile
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.tools import layer_by_layer_schema_py_generated as layer_schema_fb
from tflite_micro.tensorflow.lite.micro.tools import model_transforms_utils

np.set_printoptions(threshold=sys.maxsize)

# Usage information:
# This Python Tool/Script can first be used to compare TFLM vs Tflite outputs for
# random Inputs by only providing a TfLite file

# TfLite vs TFLM command:
#   `bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
#     --input_tflite_file=</path/to/my_model.tflite>`

# This Python Tool/Script can also be used to comapre TFLM vs Expected
# Output/Flatbuffer provided by the C++ Tool/binary.

# TFLM vs Expected Command:
#   `bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
#     --input_tflite_file=</path/to/my_model.tflite> \
#     --dbg_file=</path/to/my_debug_flatbuffer_file>`

# Optional Flags:
#   --print_dump
#           when this flags is set it will dump a part of the TFLM and Ouput
#           it's compared against for each layer

#   --rng
#           integer flag that only works in TfLite vs TFLM comparison(when only
#           a TfLite Model is Provided).It can be used to set the rng seed to a
#           differen value then it's default value of 42.

_INPUT_TFLITE_FILE = flags.DEFINE_string(
    "input_tflite_file",
    None,
    "Full path name to the input TFLite file.",
    required=True)

_RNG = flags.DEFINE_integer(
    "rng",
    42,
    "This flag defines rng seed used to generate random test data for the"
    " provided model. This only occurs when no input/golden data are provided."
    " It is defaulted to 42. ",
)

_DEBUG_FILE = flags.DEFINE_string(
    "layer_by_layer_data_file",
    None,
    "Full path to the debug file , generated in C++",
    required=False,
)

_PRINT_PREVIEW = flags.DEFINE_bool(
    "print_dump",
    False,
    "When this flag is set to True, it prints a preview of elements of the TFLM"
    " output and output it's being compared with.",
    required=False,
)


def numpy_from_tensor_type(tensor_type_idx):
  """Gives the equivalent numpy dtype based on TensorType class (schema) number."""
  tensor_type_idx_to_numpy = {
      layer_schema_fb.TensorTypes.FLOAT32:
      np.float32,
      layer_schema_fb.TensorTypes.FLOAT16:
      np.float16,
      layer_schema_fb.TensorTypes.INT32:
      np.int32,
      layer_schema_fb.TensorTypes.UINT8:
      np.uint8,
      layer_schema_fb.TensorTypes.INT64:
      np.int64,
      layer_schema_fb.TensorTypes.STRING:
      np.bytes_,
      layer_schema_fb.TensorTypes.BOOL:
      np.bool_,
      layer_schema_fb.TensorTypes.INT16:
      np.int16,
      layer_schema_fb.TensorTypes.COMPLEX64:
      np.complex64,
      layer_schema_fb.TensorTypes.INT8:
      np.int8,
      layer_schema_fb.TensorTypes.FLOAT64:
      np.float64,
      layer_schema_fb.TensorTypes.COMPLEX128:
      np.complex128,
      layer_schema_fb.TensorTypes.UINT64:
      np.uint64,
      layer_schema_fb.TensorTypes.RESOURCE:
      "RESORCE",
      layer_schema_fb.TensorTypes.VARIANT:
      "VARIANT",
      layer_schema_fb.TensorTypes.UINT32:
      np.uint32,
      layer_schema_fb.TensorTypes.UINT16:
      np.uint16,
      # INT4 is mapped to INT8, b/246806634
      layer_schema_fb.TensorTypes.INT4:
      np.int8,
  }
  return tensor_type_idx_to_numpy.get(tensor_type_idx)


def GenerateRandomInputTfLiteComparison(tflm_interpreter, tflite_interpreter,
                                        model, rng_value):
  subgraph_info = layer_schema_fb.ModelTestDataT()
  subgraph_info.subgraphData = []
  rng_seed = np.random.default_rng(seed=rng_value)

  for subgraph_index, subgraph in enumerate(model.subgraphs):
    subgraph_data = layer_schema_fb.SubgraphDataT()
    subgraph_data.subgraphIndex = subgraph_index
    subgraph_data.outputs = []

    for op_index, operator in enumerate(subgraph.operators):
      for output in operator.outputs:
        tensor_data = layer_schema_fb.TensorDataT()
        tensor_data.layerNumber = op_index
        tensor_data.tensorIndex = output
        subgraph_data.outputs.append(tensor_data)
    subgraph_info.subgraphData.append(subgraph_data)

  for index, input_tensor_index in enumerate(model.subgraphs[0].inputs):
    input_tensor = model.subgraphs[0].tensors[input_tensor_index]
    random_data = model_transforms_utils.generate_random_input_data(
        model, input_tensor, rng_seed)
    tflm_interpreter.set_input(random_data, index)
    tflite_interpreter.set_tensor(input_tensor_index, random_data)
  return subgraph_info, tflm_interpreter, tflite_interpreter


def ReadDebugFile():
  with gfile.GFile(_DEBUG_FILE.value, "rb") as debug_file_handle:
    debug_bytearray = bytearray(debug_file_handle.read())
  flatbuffer_root_object = layer_schema_fb.ModelTestData.GetRootAs(
      debug_bytearray, 0)
  debug_obj = layer_schema_fb.ModelTestDataT.InitFromObj(
      flatbuffer_root_object)
  return debug_obj


def SetDebugFileInterpreterInput(tflm_interpreter, tflite_interpreter,
                                 debug_obj):
  for inputs in debug_obj.inputData:
    input_array = np.frombuffer(bytearray(inputs.data),
                                dtype=numpy_from_tensor_type(inputs.dtype))
    input_array = np.reshape(input_array, inputs.shape)
    tflm_interpreter.set_input(input_array, inputs.inputIndex)
    tflite_interpreter.set_tensor(inputs.tensorIndex, input_array)

  return tflm_interpreter, tflite_interpreter


def main(_) -> None:
  logging.info(
      "\n--Running TFLM vs TfLite layer by layer debugger on: %s",
      _INPUT_TFLITE_FILE.value,
  )

  model = flatbuffer_utils.read_model(_INPUT_TFLITE_FILE.value)

  tflm_interpreter = runtime.Interpreter.from_file(
      _INPUT_TFLITE_FILE.value,
      intrepreter_config=runtime.InterpreterConfig.kPreserveAllTensors,
  )

  tflite_interpreter = tf.lite.Interpreter(
      model_path=_INPUT_TFLITE_FILE.value,
      experimental_preserve_all_tensors=True,
  )

  tflite_interpreter.allocate_tensors()

  debug_obj = None

  # Setting Inputs either randomly or using provided Debug File
  if _DEBUG_FILE.value == None:
    debug_obj, tflm_interpreter, tflite_interpreter = (
        GenerateRandomInputTfLiteComparison(tflm_interpreter,
                                            tflite_interpreter, model,
                                            _RNG.value))
    tflite_interpreter.invoke()
  else:
    debug_obj = ReadDebugFile()
    tflm_interpreter, tflite_interpreter = SetDebugFileInterpreterInput(
        tflm_interpreter, tflite_interpreter, debug_obj)

  tflm_interpreter.invoke()
  comparison = ""

  for subgraph in debug_obj.subgraphData:
    for output in subgraph.outputs:
      tflm_ouput = tflm_interpreter.GetTensor(
          output.tensorIndex, subgraph.subgraphIndex)["tensor_data"]

      comparison_ouput = None

      if _DEBUG_FILE.value == None:
        tflite_output = tflite_interpreter.get_tensor(output.tensorIndex,
                                                      subgraph.subgraphIndex)
        comparison_ouput = tflite_output
        comparison = "TfLite"
      else:
        expected_output_data = np.frombuffer(bytearray(output.data),
                                             dtype=numpy_from_tensor_type(
                                                 output.dtype))
        expected_output_data = np.reshape(expected_output_data, output.shape)
        comparison = "Expected Golden Data"
        comparison_ouput = expected_output_data

      error_message = (
          "\n\nTFLM output does not match {comparison} output.\n Subgraph"
          " number is {subgraph_index} \n Layer number is {layer_number} \n The"
          " Tensor Index where this output does not match is {tensor_index}"
          " \n\n\n".format(
              comparison=comparison,
              subgraph_index=subgraph.subgraphIndex,
              layer_number=output.layerNumber,
              tensor_index=output.tensorIndex,
          ))
      if _PRINT_PREVIEW.value:
        print("layer number ", output.layerNumber)
        print("tensor index ", output.tensorIndex, "\n\n")
        print("TFLM output \n ", tflm_ouput[:10])
        print(
            "{comparison} output \n".format(comparison=comparison),
            comparison_ouput[:_PRINT_PREVIEW.value],
        )
        print("--------------\n\n\n")
      np.testing.assert_array_equal(tflm_ouput,
                                    comparison_ouput,
                                    err_msg=error_message,
                                    verbose=True)
  print(
      "\n\nTFLM output matched {comparison} output for all Layers in the Model."
      .format(comparison=comparison))


if __name__ == "__main__":
  app.run(main)
