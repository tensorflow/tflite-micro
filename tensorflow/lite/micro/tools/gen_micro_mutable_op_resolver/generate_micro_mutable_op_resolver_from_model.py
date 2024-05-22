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
"""This tool generates a header with Micro Mutable Op Resolver code for a given
   model. See README.md for more info.
"""

import os
import re

from absl import app
from absl import flags
from mako import template

from tensorflow.lite.tools import visualize

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
TEMPLATE_DIR = os.path.abspath(TEMPLATE_DIR)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'common_tflite_path', None,
    'Common path to tflite files. This need to be an absolute path.'
    'This would typically be the path to the directory where the models reside.'
)
flags.DEFINE_list(
    'input_tflite_files', None,
    'Relative path name list of the input TFLite files.'
    'This would be relative to the common path.'
    'This would typically be the name(s) of the tflite file(s).')
flags.DEFINE_string('output_dir', None, 'Directory to output generated files.')
flags.DEFINE_string(
    'verify_op_list_against_header', None,
    'Take micro_mutable_op_resolver.h as input and verifies that all generated operator calls are there.'
)

flags.mark_flag_as_required('common_tflite_path')
flags.mark_flag_as_required('input_tflite_files')
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

  # Edge cases
  formated_op_string = formated_op_string.replace('Lstm', 'LSTM')
  formated_op_string = formated_op_string.replace('BatchMatmul', 'BatchMatMul')

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

  custom_op_found = False
  operators_and_activations = set()

  with open(model_path, 'rb') as f:
    data_bytes = bytearray(f.read())

  data = visualize.CreateDictFromFlatbuffer(data_bytes)

  for op_code in data["operator_codes"]:
    if op_code['custom_code'] is None:
      op_code["builtin_code"] = max(op_code["builtin_code"],
                                    op_code["deprecated_builtin_code"])
    else:
      custom_op_found = True
      operators_and_activations.add(
          visualize.NameListToString(op_code['custom_code']))

  for op_code in data["operator_codes"]:
    # Custom operator already added.
    if custom_op_found and visualize.BuiltinCodeToName(
        op_code['builtin_code']) == "CUSTOM":
      continue

    operators_and_activations.add(
        visualize.BuiltinCodeToName(op_code['builtin_code']))

  return operators_and_activations


def VerifyOpList(op_list, header):
  """Make sure operators in list are not missing in header file ."""

  supported_op_list = []
  with open(header, 'r') as f:
    for l in f.readlines():
      if "TfLiteStatus Add" in l:
        op = l.strip().split(' ')[1].split('(')[0]
        supported_op_list.append(op)

  for op in op_list:
    if op not in supported_op_list:
      print(f'{op} not supported by TFLM')
      return True

  return False


def main(_):
  model_names = []
  final_operator_list = []
  merged_operator_list = []

  common_model_path = FLAGS.common_tflite_path
  relative_model_paths = FLAGS.input_tflite_files

  for relative_model_path in relative_model_paths:
    full_model_path = f"{common_model_path}/{relative_model_path}"
    operators = GetModelOperatorsAndActivation(full_model_path)
    model_name = full_model_path.split('/')[-1]
    model_names.append(model_name)

    parsed_operator_list = []
    for op in sorted(list(operators)):
      parsed_operator_list.append(ParseString(op))

    merged_operator_list = merged_operator_list + parsed_operator_list

  number_models = len(model_names)
  if number_models > 1:
    model_name = ", ".join(model_names)

  [
      final_operator_list.append(operator) for operator in merged_operator_list
      if operator not in final_operator_list
  ]

  if FLAGS.verify_op_list_against_header and VerifyOpList(
      final_operator_list, FLAGS.verify_op_list_against_header):
    return True

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  GenerateMicroMutableOpsResolverHeaderFile(final_operator_list, model_name,
                                            FLAGS.output_dir)
  return False


if __name__ == '__main__':
  app.run(main)
