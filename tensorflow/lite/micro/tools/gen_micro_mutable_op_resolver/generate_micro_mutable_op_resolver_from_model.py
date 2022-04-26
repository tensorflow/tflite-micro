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
"""This tool generates a header with Micro Mutable Op Resolver code for a given
   model. See README.md for more info.
"""

import os
import re

from absl import app
from absl import flags
from mako import template

from tflite_micro.tensorflow.lite.tools import visualize as visualize

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
TEMPLATE_DIR = os.path.abspath(TEMPLATE_DIR)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input TFLite file.')
flags.DEFINE_string('output_dir', None, 'directory to output generated files')

flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_dir')


def ParseString(word):
  """Converts a flatbuffer operator string to a format suitable for Micro
     Mutable Op Resolver. Example: CONV_2D --> AddConv2D."""

  # Edge case for AddDetectionPostprocess().
  # The custom code is TFLite_Detection_PostProcess.
  word = word.replace('TFLite', '')

  word_split = re.split('_|-', word)
  formated_op_string = ''
  for part in word_split:
    if len(part) > 1:
      if part[0].isalpha():
        formated_op_string += part[0].upper() + part[1:].lower()
      else:
        formated_op_string += part.upper()
    else:
      formated_op_string += part.upper()
  return 'Add' + formated_op_string


def GenerateMicroMutableOpsResolverHeaderFile(operators, name_of_model,
                                              output_dir):
  """Generates Micro Mutable Op Resolver code based on a template."""

  number_of_ops = len(operators)
  outfile = 'micro_mutable_op_resolver.h'

  template_file_path = os.path.join(TEMPLATE_DIR, outfile + '.mako')
  build_template = template.Template(filename=template_file_path)
  with open(output_dir + '/gen_' + outfile, 'w') as file_obj:
    key_values_in_template = {
        'model': name_of_model,
        'number_of_ops': number_of_ops,
        'operators': operators
    }
    file_obj.write(build_template.render(**key_values_in_template))


def GetModelOperatorsAndActivation(model_path):
  """Extracts a set of operators from a tflite model."""

  operators_and_activations = set()

  with open(model_path, 'rb') as f:
    data_bytes = bytearray(f.read())

  data = visualize.CreateDictFromFlatbuffer(data_bytes)

  for op_code in data["operator_codes"]:
    if op_code['custom_code'] is None:
      op_code["builtin_code"] = max(op_code["builtin_code"],
                                    op_code["deprecated_builtin_code"])
    else:
      operators_and_activations.add(
          visualize.NameListToString(op_code['custom_code']))

  for op_code in data["operator_codes"]:
    operators_and_activations.add(
        visualize.BuiltinCodeToName(op_code['builtin_code']))

  return operators_and_activations


def main(_):
  model_path = FLAGS.input_tflite_file
  operators = GetModelOperatorsAndActivation(model_path)
  model_name = model_path.split('/')[-1]
  base_model_name = model_name.split('.')[0]
  out_dir = FLAGS.output_dir + '/' + base_model_name

  parsed_operator_list = []
  for op in sorted(list(operators)):
    parsed_operator_list.append(ParseString(op))

  os.makedirs(out_dir, exist_ok=True)
  GenerateMicroMutableOpsResolverHeaderFile(parsed_operator_list, model_name,
                                            out_dir)


if __name__ == '__main__':
  app.run(main)
