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
# =============================================================================
"""Simple TF model creation using resource variables.

Model is built either with basic TF functions (concrete function model), or via
Keras. The model simply mimics an accumulator (via the persistent memory / state
functionality of resource variables), taking in two inputs:
1) A boolean for choosing addition/subtraction.
2) The value to add/subtract from the accumulator variable.


Useful links:
https://www.tensorflow.org/lite/models/convert/convert_models#convert_concrete_functions_
https://www.tensorflow.org/guide/function#creating_tfvariables
https://www.tensorflow.org/api_docs/python/tf/Variable
https://www.tensorflow.org/api_docs/python/tf/function
"""

import os


def get_model_from_keras():
  """Accumulator model loaded from pre-generated tflite file."""
  model_path = os.path.join(
    os.path.dirname(__file__), "resource_variables.tflite"
  )
  with open(model_path, "rb") as f:
    return f.read()


def get_model_from_concrete_function():
  """Accumulator model loaded from pre-generated tflite file."""
  return get_model_from_keras()

