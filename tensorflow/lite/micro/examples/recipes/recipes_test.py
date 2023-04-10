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
#
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.examples.recipes import resource_variables_lib
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


class RecipesTest(test_util.TensorFlowTestCase):

  def _use_tflite_interpreter(self, filename, input_data_list):
    tflite_interpreter = tf.lite.Interpreter(
        model_path=filename,
        # TODO(b/): Latest TF pypi release package does not include registerer
        # change needed to use BUILTIN_REF(CALL_ONCE op). Update once included.
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO,
    )
    tflite_interpreter.allocate_tensors()
    output_list = []
    for input_dataset_index in range(3):
      for idx, input_details in enumerate(
          tflite_interpreter.get_input_details()
      ):
        input_tensor_idx = input_details["index"]
        input_data = np.reshape(
            input_data_list[input_dataset_index][idx], input_details["shape"]
        )
        tflite_interpreter.set_tensor(input_tensor_idx, input_data)

      output_tensor_idx = tflite_interpreter.get_output_details()[0]["index"]
      tflite_interpreter.invoke()
      output_list.append(tflite_interpreter.get_tensor(output_tensor_idx))

    tflite_interpreter.reset_all_variables()
    for idx, input_details in enumerate(tflite_interpreter.get_input_details()):
      input_tensor_idx = input_details["index"]
      input_data = np.reshape(input_data_list[3][idx], input_details["shape"])
      tflite_interpreter.set_tensor(input_tensor_idx, input_data)

    output_tensor_idx = tflite_interpreter.get_output_details()[0]["index"]
    tflite_interpreter.invoke()
    output_list.append(tflite_interpreter.get_tensor(output_tensor_idx))
    return output_list

  def _use_tflm_interpreter(self, filename, input_data_list):
    tflm_interpreter = tflm_runtime.Interpreter.from_file(filename)
    output_list = []
    for input_dataset_index in range(3):
      for idx, input_data in enumerate(input_data_list[input_dataset_index]):
        input_data = np.reshape(
            input_data, tflm_interpreter.get_input_details(idx).get("shape")
        )
        tflm_interpreter.set_input(input_data, idx)
      tflm_interpreter.invoke()
      output_list.append(tflm_interpreter.get_output(0))

    tflm_interpreter.reset()
    for idx, input_data in enumerate(input_data_list[3]):
      input_data = np.reshape(
          input_data, tflm_interpreter.get_input_details(idx).get("shape")
      )
      tflm_interpreter.set_input(input_data, idx)
    tflm_interpreter.invoke()
    output_list.append(tflm_interpreter.get_output(0))
    return output_list

  def test_resource_variables_model(self):
    use_concrete = False
    test_dir = self.get_temp_dir()
    model_filename = test_dir + "/resource_var_model.tflite"
    tf_model = resource_variables_lib.get_tf_model(use_concrete)
    resource_variables_lib.convert_and_save_model(
        tf_model, use_concrete, model_filename
    )

    output_goldens_tflm = [5.0, 10.0, 5.0, 75.0]
    # TODO(b/): Lite does not reset resource variables to default value.
    output_goldens_lite = [5.0, 10.0, 5.0, 80.0]
    accumulator_input = [np.full((100,), 5.0, dtype=np.float32)]
    input_data_list = [
        [[True], accumulator_input],
        [[True], accumulator_input],
        [[False], accumulator_input],
        [[True], [np.full((100,), 75.0, dtype=np.float32)]],
    ]
    tflite_out_list = self._use_tflite_interpreter(
        model_filename, input_data_list
    )
    tflm_out_list = self._use_tflm_interpreter(model_filename, input_data_list)

    for tflite_out, tflm_out, output_expected_tflm, output_expected_lite in zip(
        tflite_out_list, tflm_out_list, output_goldens_tflm, output_goldens_lite
    ):
      self.assertAllEqual(
          tflite_out.flatten(),
          np.full((100), output_expected_lite, dtype=np.float32),
      )
      self.assertAllEqual(
          tflm_out.flatten(),
          np.full((100), output_expected_tflm, dtype=np.float32),
      )


if __name__ == "__main__":
  test.main()
