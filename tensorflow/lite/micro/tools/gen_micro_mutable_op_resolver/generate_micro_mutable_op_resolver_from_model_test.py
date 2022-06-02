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

import os
import shutil

from absl import app
from absl import flags
from mako import template
from tflite_micro.tensorflow.lite.micro.tools import generate_test_for_model

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
TEMPLATE_DIR = os.path.abspath(TEMPLATE_DIR)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input TFLite file.')
flags.DEFINE_string(
    'output_dir', None, 'Directory to output generated files. \
  Note that final output will be in FLAGS.output_dir/<base name of model>. \
  Where <base name of model> will come from FLAGS.input_tflite_file.')
flags.DEFINE_integer('arena_size', 1024 * 136, 'Size of arena')
flags.DEFINE_boolean('verify_output', False,
                     'Verify output or just run model.')

flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_dir')


class MicroMutableOpTestGenerator(generate_test_for_model.TestDataGenerator):

  def __init__(self, output_dir, model_path, verify_output, arena_size):
    super().__init__(output_dir, [model_path], [0])  # Third argument not used.
    self.verify_output = verify_output
    self.arena_size = arena_size

    self.target = model_path.split('/')[-1].split('.')[0]
    self.target_with_path = model_path.split('tflite_micro/')[-1]. \
        split('tflite-micro/')[-1].split('.')[0]

    # Only int8 models supported
    self.input_type = 'int8'
    self.output_type = 'int8'
    self.input_types = [self.input_type]

  def generate_golden(self):
    if not self.verify_output:
      return
    super().generate_golden_single_in_single_out()

  def generate_test(self, template_dir, template_file, out_file):
    template_file_path = os.path.join(template_dir, template_file)
    build_template = template.Template(filename=template_file_path)
    path_to_target = self.target_with_path.split('/' + self.target)[0] + \
        '/' + self.target
    with open(self.output_dir + '/' + out_file, 'w') as file_obj:
      key_values_in_template = {
          'arena_size': self.arena_size,
          'verify_output': int(self.verify_output),
          'path_to_target': path_to_target,
          'target': self.target,
          'target_with_path': self.target_with_path,
          'input_dtype': self.input_type,
          'output_dtype': self.output_type
      }
      file_obj.write(build_template.render(**key_values_in_template))

  def generate_build_file(self, template_dir):
    template_file_path = os.path.join(template_dir, 'BUILD.mako')
    build_template = template.Template(filename=template_file_path)
    with open(self.output_dir + '/BUILD', 'w') as file_obj:
      key_values_in_template = {
          'verify_output': self.verify_output,
          'target': self.target,
          'input_dtype': self.input_type,
          'output_dtype': self.output_type
      }
      file_obj.write(build_template.render(**key_values_in_template))


def main(_):
  model_path = FLAGS.input_tflite_file
  model_name = model_path.split('/')[-1]
  base_model_name = model_name.split('.')[0]
  name_of_make_target = 'generated_micro_mutable_op_resolver_' + base_model_name

  out_dir = FLAGS.output_dir + '/' + base_model_name
  os.makedirs(out_dir, exist_ok=True)

  # Copy model to out dir to get the Mako generation right
  new_model_path = out_dir + '/' + model_name
  shutil.copyfile(model_path, new_model_path)

  data_generator = MicroMutableOpTestGenerator(out_dir, new_model_path,
                                               FLAGS.verify_output,
                                               FLAGS.arena_size)
  data_generator.generate_golden()
  data_generator.generate_build_file(TEMPLATE_DIR)
  data_generator.generate_makefile(
      test_file='micro_mutable_op_resolver_test.cc',
      src_prefix=name_of_make_target)
  data_generator.generate_test(
      TEMPLATE_DIR,
      template_file='micro_mutable_op_resolver_test.cc.mako',
      out_file='micro_mutable_op_resolver_test.cc')


if __name__ == '__main__':
  app.run(main)
