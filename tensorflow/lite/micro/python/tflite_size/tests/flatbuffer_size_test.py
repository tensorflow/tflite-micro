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
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.python.tflite_size.src import flatbuffer_size

root_dir = 'tensorflow/lite/micro/python/tflite_size/tests'


class FlatbufferSizeTest(test_util.TensorFlowTestCase):

  def _compareFile(self, file1, file2):
    with open(file1, 'rb') as f1:
      data1 = f1.read()
    with open(file2, 'rb') as f2:
      data2 = f2.read()
    self.assertEqual(data1, data2)

  def testCompareWithTFLite(self):
    in_filename = root_dir + '/simple_add_model.tflite'
    out_json_file = root_dir + '/simple_add_model.json'
    out_html_file = root_dir + '/simple_add_model.json.html'
    gold_json_file = root_dir + '/gold_simple_add_model_json.txt'
    gold_html_file = root_dir + '/gold_simple_add_model_html.txt'

    flatbuffer_size.convert_tflite_to_html(in_filename, out_html_file,
                                           out_json_file)

    self._compareFile(out_json_file, gold_json_file)
    self._compareFile(out_html_file, gold_html_file)


if __name__ == '__main__':
  test.main()
