# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite Python Interface: Sanity check."""
import os
import re

from tflite_micro.tensorflow.lite.tools import test_utils
from tflite_micro.tensorflow.lite.tools import visualize
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class VisualizeTest(test_util.TensorFlowTestCase):

  def testTensorTypeToName(self):
    self.assertEqual('FLOAT32', visualize.TensorTypeToName(0))

  def testBuiltinCodeToName(self):
    self.assertEqual('HASHTABLE_LOOKUP', visualize.BuiltinCodeToName(10))

  def testFlatbufferToDict(self):
    model = test_utils.build_mock_flatbuffer_model()
    model_dict = visualize.CreateDictFromFlatbuffer(model)
    self.assertEqual(test_utils.TFLITE_SCHEMA_VERSION, model_dict['version'])
    self.assertEqual(1, len(model_dict['subgraphs']))
    self.assertEqual(2, len(model_dict['operator_codes']))
    self.assertEqual(3, len(model_dict['buffers']))
    self.assertEqual(3, len(model_dict['subgraphs'][0]['tensors']))
    self.assertEqual(0, model_dict['subgraphs'][0]['tensors'][0]['buffer'])

  def testVisualize(self):
    model = test_utils.build_mock_flatbuffer_model()
    tmp_dir = self.get_temp_dir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')
    with open(model_filename, 'wb') as model_file:
      model_file.write(model)

    html_text = visualize.create_html(model_filename)

    # It's hard to test debug output without doing a full HTML parse,
    # but at least sanity check that expected identifiers are present.
    self.assertRegex(
        html_text, re.compile(r'%s' % model_filename, re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text,
                     re.compile(r'input_tensor', re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text,
                     re.compile(r'constant_tensor', re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text, re.compile(r'ADD', re.MULTILINE | re.DOTALL))


if __name__ == '__main__':
  test.main()
