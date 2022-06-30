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


class FNode:
  """ A node representing a flatbuffer element. """

  def __init__(self):
    self.isLeaf = False
    self.name = ''
    self.children = list()
    self.size = 0
    self.value = 0

  def print(self):
    print("%d, %s, %d, %d, %s" %
          (self.isLeaf, self.name, len(self.children), self.size, self.value))


"""
FlatbufferSizeGraph converts a flatbuffer in json string (with size info ) into a graph of nodes. 

A basic node structure corresponds to the following json string: 

field_name: {value: xxxx, total_size: }
field_name: {value: [ ], total_size: }

where value can be:
1. a dict (a new structure) that is not just value and total_size
2. a list (a new array)
3. a scalar (neither dict nor list). 
"""


class FlatbufferSizeGraph:

  def __init__(self):
    self._root = FNode()
    self._verbose = False

  def _buildNodeForField(self, name, flatbuffer_json):
    fNode = FNode()
    fNode.name = name

    if self._verbose:
      print("Start processing %s" % flatbuffer_json)
      fNode.print()

    if "value" in flatbuffer_json.keys(
    ) and "total_size" in flatbuffer_json.keys():
      fNode.size = flatbuffer_json["total_size"]
      self._processValue(fNode, flatbuffer_json["value"])
    else:
      raise Exception("Filed not a dict with value and total size!")

    if self._verbose:
      print("End processing %s" % flatbuffer_json)
      fNode.print()
    return fNode

  def _processValue(self, fNode, value_in_flatbuffer_json):
    if type(value_in_flatbuffer_json) is not dict and type(
        value_in_flatbuffer_json) is not list:
      fNode.value = value_in_flatbuffer_json
      fNode.isLeaf = True

    if type(value_in_flatbuffer_json) is dict:
      if "value" in value_in_flatbuffer_json.keys(
      ) and "total_size" in value_in_flatbuffer_json.keys():
        raise Exception(
            "Field is another dict with value and total size again??")

      for name in value_in_flatbuffer_json.keys():
        fNode.children.append(
            self._buildNodeForField(name, value_in_flatbuffer_json[name]))
    elif type(value_in_flatbuffer_json) is list:
      for nidx, next_obj in enumerate(value_in_flatbuffer_json):
        leaf_name = "%s[%d]" % (fNode.name, nidx)
        # array: "operator_codes": {"value": [{"value": {"version": {"value": 2, "total_size": 4}}, "total_size": 4}], "total_size": 4}
        # so, each element must be of {"value: { field_name: }", total_size}
        fNode.children.append(self._buildNodeForField(leaf_name, next_obj))

  def createGraph(self, flatbuffer_in_json_with_size):
    self._root = self._buildNodeForField("ROOT", flatbuffer_in_json_with_size)

  def displayGraph(self, graphTraveser):
    return graphTraveser.displayFlatbuffer(self._root)
