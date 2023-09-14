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
""" Operator Constants """

from typing import Dict

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb

ACTIVATION_FUNCS: Dict[int, str] = {
    schema_fb.ActivationFunctionType.NONE: "kTfLiteActNone",
    schema_fb.ActivationFunctionType.RELU: "kTfLiteActRelu",
    schema_fb.ActivationFunctionType.RELU_N1_TO_1: "kTfLiteActReluN1To1",
    schema_fb.ActivationFunctionType.RELU6: "kTfLiteActRelu6",
    schema_fb.ActivationFunctionType.TANH: "kTfLiteActTanh",
    schema_fb.ActivationFunctionType.SIGN_BIT: "kTfLiteActSignBit",
}

TFLITE_TYPE: Dict[int, str] = {
    0: "kTfLiteNoType",
    1: "kTfLiteFloat32",
    2: "kTfLiteInt32",
    3: "kTfLiteUInt8",
    4: "kTfLiteInt64",
    5: "kTfLiteString",
    6: "kTfLiteBool",
    7: "kTfLiteInt16",
    8: "kTfLiteComplex64",
    9: "kTfLiteInt8",
    10: "kTfLiteFloat16",
    11: "kTfLiteFloat64",
    12: "kTfLiteComplex128",
    13: "kTfLiteUInt64",
    14: "kTfLiteResource",
    15: "kTfLiteVariant",
    16: "kTfLiteUInt32",
    17: "kTfLiteUInt16",
    18: "kTfLiteInt4",
}
