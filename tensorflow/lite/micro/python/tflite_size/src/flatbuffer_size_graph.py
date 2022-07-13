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

  def _build_node_for_field(self, name, flatbuffer_json):
    node = FNode()
    node.name = name

    if self._verbose:
      print("Start processing %s" % flatbuffer_json)
      node.print()

    if "value" in flatbuffer_json.keys(
    ) and "total_size" in flatbuffer_json.keys():
      node.size = flatbuffer_json["total_size"]
      self._process_value(node, flatbuffer_json["value"])
    else:
      raise Exception("Filed not a dict with value and total size!")

    if self._verbose:
      print("End processing %s" % flatbuffer_json)
      node.print()
    return node

  def _process_value(self, node, value_in_flatbuffer_json):
    if type(value_in_flatbuffer_json) is not dict and type(
        value_in_flatbuffer_json) is not list:
      node.value = value_in_flatbuffer_json
      node.isLeaf = True

    if type(value_in_flatbuffer_json) is dict:
      if "value" in value_in_flatbuffer_json.keys(
      ) and "total_size" in value_in_flatbuffer_json.keys():
        raise Exception(
            "Field is another dict with value and total size again??")

      for name in value_in_flatbuffer_json.keys():
        node.children.append(
            self._build_node_for_field(name, value_in_flatbuffer_json[name]))
    elif type(value_in_flatbuffer_json) is list:
      for nidx, next_obj in enumerate(value_in_flatbuffer_json):
        leaf_name = "%s[%d]" % (node.name, nidx)
        # array: "operator_codes": {"value": [{"value": {"version": {"value": 2, "total_size": 4}}, "total_size": 4}], "total_size": 4}
        # so, each element must be of {"value: { field_name: }", total_size}
        node.children.append(self._build_node_for_field(leaf_name, next_obj))

  def create_graph(self, flatbuffer_in_json_with_size):
    self._root = self._build_node_for_field("ROOT",
                                            flatbuffer_in_json_with_size)

  def display_graph(self, graph_traveser):
    return graph_traveser.display_flatbuffer(self._root)
