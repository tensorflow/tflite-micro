# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

import json
import unittest

from tflite_micro.tensorflow.lite.micro.compression.tutorial import md_to_ipynb


def _convert(*lines):
  """Returns the notebook converted from the markdown lines, as a dict."""
  return json.loads(md_to_ipynb.convert("\n".join(lines) + "\n"))


def _cells(*lines):
  """Returns (cell_type, source) pairs for the markdown lines."""
  return [
    (cell["cell_type"], "".join(cell["source"]))
    for cell in _convert(*lines)["cells"]
  ]


class MetadataTest(unittest.TestCase):
  def test_header_jupyter_mapping_becomes_metadata(self):
    notebook = _convert(
      "---",
      "jupyter:",
      "  jupytext:",
      "    text_representation:",
      "      extension: .md",
      "  kernelspec:",
      "    language: python",
      "    name: python3",
      "---",
      "",
      "text",
    )
    self.assertEqual(
      notebook["metadata"],
      {"kernelspec": {"language": "python", "name": "python3"}},
    )
    self.assertEqual([c["cell_type"] for c in notebook["cells"]], ["markdown"])

  def test_no_header_gives_empty_metadata(self):
    self.assertEqual(_convert("text")["metadata"], {})


class CellTest(unittest.TestCase):
  def test_python_block_becomes_code_cell(self):
    self.assertEqual(
      _cells("before", "", "```python", "x = 1", "", "y = 2", "```", "after"),
      [
        ("markdown", "before"),
        ("code", "x = 1\n\ny = 2"),
        ("markdown", "after"),
      ],
    )

  def test_code_cell_has_no_outputs(self):
    (cell,) = _convert("```python", "x = 1", "```")["cells"]
    self.assertEqual(cell["outputs"], [])
    self.assertIsNone(cell["execution_count"])

  def test_two_blank_lines_split_markdown(self):
    self.assertEqual(
      _cells("first", "", "", "second"),
      [("markdown", "first"), ("markdown", "second")],
    )

  def test_one_blank_line_does_not_split_markdown(self):
    self.assertEqual(
      _cells("first", "", "second"),
      [("markdown", "first\n\nsecond")],
    )

  def test_blank_lines_inside_other_fence_do_not_split(self):
    self.assertEqual(
      _cells("```yaml", "a: 1", "", "", "b: 2", "```"),
      [("markdown", "```yaml\na: 1\n\n\nb: 2\n```")],
    )

  def test_python_line_inside_other_fence_stays_markdown(self):
    self.assertEqual(
      _cells("```", "```python", "x = 1", "```"),
      [("markdown", "```\n```python\nx = 1\n```")],
    )

  def test_cell_ids_follow_document_order(self):
    cells = _convert("one", "", "", "two", "```python", "three", "```")["cells"]
    self.assertEqual([c["id"] for c in cells], ["cell-0", "cell-1", "cell-2"])


class ErrorTest(unittest.TestCase):
  def test_unclosed_header_names_its_line(self):
    with self.assertRaisesRegex(ValueError, "YAML header .* line 1 "):
      _convert("---", "jupyter: {}", "text")

  def test_unclosed_code_block_names_its_line_in_the_file(self):
    with self.assertRaisesRegex(ValueError, "python block .* line 5 "):
      _convert("---", "jupyter: {}", "---", "text", "```python", "x = 1")


if __name__ == "__main__":
  unittest.main()
