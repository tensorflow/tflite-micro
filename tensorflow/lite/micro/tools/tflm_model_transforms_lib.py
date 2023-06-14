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
"""TFLM specific flatbuffer model transformations, to reduce model size.

go/tflm-flatbuffer-reduction
We take advantage of the TFLM infrastructure to remove information in the
flatbuffer which we do not preciscely need for inference of a model.
The methods used here require the assumptions made from the TFLM framework to
properly work.
"""

import os
import tempfile
from absl import logging
import numpy as np
from tensorflow.python.platform import gfile

from tflite_micro.tensorflow.lite.tools import flatbuffer_utils
from tflite_micro.tensorflow.lite.micro.tools import model_transforms_utils
from tflite_micro.python.tflite_micro import runtime


def _save_and_align_flatbuffer(model, model_path):
  flatbuffer_utils.write_model(model, model_path)
  model_transforms_utils.tflite_flatbuffer_align(model_path, model_path)


def log_size_difference(input_path, transformed_model_path):
  initial_binary_size = gfile.Stat(input_path).length
  final_binary_size = gfile.Stat(transformed_model_path).length
  logging.info("Initial file size: %d %s", initial_binary_size, "bytes.")
  logging.info("Final file size: %d %s", final_binary_size, "bytes.")
  logging.info("Savings = %d %s", initial_binary_size - final_binary_size,
               "bytes.")
  logging.info(
      " (%.2f %s",
      round((1 - (final_binary_size / initial_binary_size)) * 100, 2),
      "% reduction )",
  )


def check_models_equivalent(initial_model_path: str = None,
                            secondary_model_path: str = None,
                            test_vector_count: int = 1,
                            seed: int = 42,
                            custom_op_registerers=[]):
  """Checks that the two models are equivalent by testing that the same set of random inputs produce the same outputs using the TFLM interpreter.

  Note that this function does not test the correctness of the inference. It
  only serves to confirm that the two models are equivalent.
  The dimensions of the models inputs and outputs must be identical.

  Args:
    initial_model_path: first model full path (str)
    secondary_model_path: second model full path (str)
    test_vector_count: number of different (random) input vectors to use to test
      for equivalence.
    seed: optionally provide a custom seed value for random number generator
    custom_op_registerers: if your model makes use of custom ops

  Raises:
    AssertionError if outputs of TFLM invocations are not equal
  """
  with gfile.Open(initial_model_path, "rb") as input_model_file:
    initial_model_interpreter = runtime.Interpreter.from_bytes(
        input_model_file.read(),
        custom_op_registerers=custom_op_registerers,
    )

  with gfile.Open(secondary_model_path, "rb") as secondary_model_file:
    secondary_model_interpreter = runtime.Interpreter.from_bytes(
        secondary_model_file.read(),
        custom_op_registerers=custom_op_registerers,
    )

  initial_model_object = flatbuffer_utils.read_model(initial_model_path)
  rng = np.random.default_rng(seed=seed)

  for _ in range(test_vector_count):
    for idx, input_tensor_idx in enumerate(
        initial_model_object.subgraphs[0].inputs):
      input_tensor = initial_model_object.subgraphs[0].tensors[
          input_tensor_idx]
      rand_data = model_transforms_utils.generate_random_input_data(
          initial_model_object, input_tensor, rng)
      initial_model_interpreter.set_input(rand_data, idx)
      secondary_model_interpreter.set_input(rand_data, idx)

    initial_model_interpreter.invoke()
    secondary_model_interpreter.invoke()

    for idx, _ in enumerate(initial_model_object.subgraphs[0].outputs):
      np.testing.assert_array_equal(
          initial_model_interpreter.get_output(idx),
          secondary_model_interpreter.get_output(idx),
      )

    initial_model_interpreter.reset()
    secondary_model_interpreter.reset()


def apply_transform_and_log(
    transform_func,
    model,
    log_string,
    save_model,
    output_dir,
    filepath,
):
  """Calls transform_func(model) and logs transformed model to output_dir/filepath.

  Args:
    transform_func: the transformation function to apply
    model: tflite flatbuffer model
    log_string: information string about the transformation
    save_model: boolean whether to save the model
    output_dir: directory to write the model to
    filepath: name to save the model as

  Returns:
    transformed model object
  """
  logging.info("Applying transform: %s", log_string)
  transform_func(model)
  if not save_model:
    return model
  output_path = os.path.join(output_dir, filepath)
  _save_and_align_flatbuffer(model, output_path)
  logging.info("Output of this transform located at: %s", output_path)
  return model


def run_all_transformations(
    input_path,
    transformed_model_path,
    save_intermediates=False,
    test_transformed_model=True,
    custom_save_dir=None,
    custom_op_registerers=[],
):
  """Apply all current transform methods on an input .tflite file, and optionally save the models between methods.

  Args:
    input_path: the input .tflite model path
    transformed_model_path: output model path if not saving intermediates.
    save_intermediates: whether to save intermediate models to a tmp folder
    test_transformed_model: optional flag to enable/disable testing of
      input/transformed models on random data
    custom_save_dir: optionally pass the directory path for saving files
    custom_op_registerers: if your model makes use of custom ops

  Raises:
    AssertionError if outputs of TFLM invocations on input and transformed
    models are not equal
  """
  output_dir = None
  # We only use output_dir for the case of saving intermediate models
  if save_intermediates:
    output_dir = custom_save_dir or tempfile.mkdtemp()
    logging.info("Saving models to: %s", output_dir)

  model = flatbuffer_utils.read_model(input_path)
  pre_transform_model_path = input_path

  transforms_list = [
      model_transforms_utils.clear_resource_variable_buffers,
      model_transforms_utils.remove_extraneous_quantization_data,
      flatbuffer_utils.strip_strings,
      model_transforms_utils.shorten_variable_shared_names,
  ]
  transform_names = [
      "Clear Resource Variable Buffers",
      "Remove Extra Quantization Data",
      "Strip Strings",
      "Shorten Variable Shared Names",
  ]
  intermediate_file_names = [
      "resource_buffer_cleared.tflite",
      "quant_data_removed.tflite",
      "string_stripped.tflite",
      "variable_shared_names_shortened.tflite",
  ]

  for transform, name, file_name in zip(transforms_list, transform_names,
                                        intermediate_file_names):
    model = apply_transform_and_log(transform, model, name, save_intermediates,
                                    output_dir, file_name)

    # Testing will only work if the file has been saved to output path.
    # The "final" stage of a transformation is after it has been flatbuffer
    # aligned, hence this function only works on file paths, instead of objects.
    if test_transformed_model and save_intermediates:
      output_path = os.path.join(output_dir, file_name)
      check_models_equivalent(
          initial_model_path=pre_transform_model_path,
          secondary_model_path=output_path,
          custom_op_registerers=custom_op_registerers,
      )
      pre_transform_model_path = output_path

  gfile.MakeDirs(os.path.dirname(transformed_model_path))
  _save_and_align_flatbuffer(model, transformed_model_path)
  logging.info("Transformed model located at: %s", transformed_model_path)

  if test_transformed_model:
    check_models_equivalent(
        initial_model_path=input_path,
        secondary_model_path=transformed_model_path,
        custom_op_registerers=custom_op_registerers,
    )

  log_size_difference(input_path, transformed_model_path)
