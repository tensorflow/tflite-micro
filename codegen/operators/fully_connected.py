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
""" FullyConnected operator """

from typing import Dict
import string

from tflite_micro.codegen.operators import constants
from tflite_micro.codegen.operators import operator
from tflite_micro.codegen import utils
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb

_WEIGHTS_FORMATS: Dict[int, str] = {
    schema_fb.FullyConnectedOptionsWeightsFormat.DEFAULT:
    "kTfLiteFullyConnectedWeightsFormatDefault",
    schema_fb.FullyConnectedOptionsWeightsFormat.SHUFFLED4x16INT8:
    "kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8",
}


class FullyConnected(operator.Operator):

  def __init__(self, op: schema_fb.OperatorT):
    assert op.builtinOptionsType == schema_fb.BuiltinOptions.FullyConnectedOptions
    super(FullyConnected, self).__init__(op)
    self._builtin_options: schema_fb.FullyConnectedOptionsT = op.builtinOptions

  def generate_c_builtin_data(self) -> str:
    builtin_template = string.Template(
        "TfLiteFullyConnectedParams builtin_data = {\n"
        "    .activation = ${activation},\n"
        "    .weights_format = ${weights_format},\n"
        "    .keep_num_dims = ${keep_num_dims},\n"
        "    .asymmetric_quantize_inputs = ${asymmetric_quantize_inputs},\n"
        "    .quantized_bias_type = ${quantized_bias_type}};")
    return builtin_template.substitute(
        activation=constants.ACTIVATION_FUNCS[
            self._builtin_options.fusedActivationFunction],
        weights_format=_WEIGHTS_FORMATS[self._builtin_options.weightsFormat],
        keep_num_dims=utils.bool_to_c_str(self._builtin_options.keepNumDims),
        asymmetric_quantize_inputs=utils.bool_to_c_str(
            self._builtin_options.asymmetricQuantizeInputs),
        quantized_bias_type=constants.TFLITE_TYPE[
            self._builtin_options.quantizedBiasType])
