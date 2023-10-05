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
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import runtime
from tflite_micro.tensorflow.lite.micro.tools import model_transforms_utils

np.set_printoptions(threshold=sys.maxsize)

# Usage information:
# Default no Golden Data/Input provided will Compare TFLM vs TfLite using random input:
#   `bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
#     --input_tflite_file=</path/to/my_model.tflite>`

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
  """
  subgraph_info
  Dictionary with the following format:
  subgraph index  :
        {
             layer_number/op_index : [output_tensors] for corresponding layer
        }
  """

  # TODO(b/302738798): look into refactoring subgraph info this based on
  # Comments RJ made regarding turning this into a list of a defined
  # OutputTensor class that has relevant info

  subgraph_info = {}
  rng_seed = np.random.default_rng(seed=_RNG.value)

  for subgraph_index, subgraph in enumerate(model.subgraphs):
    subgraph_info[subgraph_index] = {}

    for op_index, operator in enumerate(subgraph.operators):
      subgraph_info[subgraph_index][op_index] = operator.outputs

  for index, input_tensor_index in enumerate(model.subgraphs[0].inputs):
    input_tensor = model.subgraphs[0].tensors[input_tensor_index]
    random_data = model_transforms_utils.generate_random_input_data(
        model, input_tensor, rng_seed)
    tflm_interpreter.set_input(random_data, index)
    tflite_interpreter.set_tensor(input_tensor_index, random_data)

  tflm_interpreter.invoke()
  tflite_interpreter.invoke()

  for subgraph_index in range(len(subgraph_info)):
    for layer_number in range(len(subgraph_info[subgraph_index])):
      tflm_ouput = tflm_interpreter.GetTensor(
          subgraph_info[subgraph_index][layer_number][0],
          subgraph_index)["tensor_data"]
      tflite_output = tflite_interpreter.get_tensor(
          subgraph_info[subgraph_index][layer_number][0], subgraph_index)
      error_message = (
          "\n\nTFLM output does not match TfLite output.\n Subgraph number is"
          " {subgraph_index} \n Layer number is {layer_number} \n The Tensor"
          " Index where this output does not match is {tensor_index} \n\n\n".
          format(
              subgraph_index=subgraph_index,
              layer_number=layer_number,
              tensor_index=subgraph_info[subgraph_index][layer_number][0],
          ))
      np.testing.assert_array_equal(tflm_ouput,
                                    tflite_output,
                                    err_msg=error_message,
                                    verbose=True)
  print("\n\nTFLM output matched TfLite output for all Layers in the Model.")


if __name__ == "__main__":
  app.run(main)
