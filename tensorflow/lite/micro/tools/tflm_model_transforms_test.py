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
"""Testing for the tflm_model_transforms functions.

Applies all transforms on various models, and uses
check_models_equivalent() to assert results.
"""
import os

from absl.testing import parameterized
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

from tflite_micro.tensorflow.lite.micro.tools import tflm_model_transforms_lib
from tflite_micro.tensorflow.lite.micro.examples.recipes import resource_variables_lib
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils


class TflmModelTransformsTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  @parameterized.named_parameters(
      ("person_detect", "person_detect.tflite"),
      ("keyword_scrambled", "keyword_scrambled.tflite"),
  )
  def test_model_transforms(self, input_file_name):
    test_tmpdir = self.get_temp_dir()
    prefix_path = resource_loader.get_path_to_datafile("../models")
    input_file_name = os.path.join(prefix_path, input_file_name)
    transformed_model_path = test_tmpdir + "/transformed.tflite"

    tflm_model_transforms_lib.run_all_transformations(
        input_path=input_file_name,
        transformed_model_path=transformed_model_path,
        save_intermediates=True,
        test_transformed_model=True,
        custom_save_dir=test_tmpdir)

    tflm_model_transforms_lib.check_models_equivalent(
        initial_model_path=input_file_name,
        secondary_model_path=transformed_model_path,
        test_vector_count=5,
    )

  # TODO(b/274635545): refactor functions to take in flatbuffer objects instead
  # of writing to files here
  def test_resource_model(self):
    test_tmpdir = self.get_temp_dir()
    resource_model = resource_variables_lib.get_model_from_keras()
    input_file_name = test_tmpdir + "/resource.tflite"
    flatbuffer_utils.write_model(
        flatbuffer_utils.convert_bytearray_to_object(resource_model),
        input_file_name)
    transformed_model_path = test_tmpdir + "/transformed.tflite"

    tflm_model_transforms_lib.run_all_transformations(
        input_path=input_file_name,
        transformed_model_path=transformed_model_path,
        save_intermediates=True,
        test_transformed_model=True,
        custom_save_dir=test_tmpdir)

    tflm_model_transforms_lib.check_models_equivalent(
        initial_model_path=input_file_name,
        secondary_model_path=transformed_model_path,
        test_vector_count=5,
    )


if __name__ == "__main__":
  test.main()
