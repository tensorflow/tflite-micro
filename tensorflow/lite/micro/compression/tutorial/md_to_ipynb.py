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
"""Converts a markdown notebook to a Jupyter .ipynb file.

The markdown uses the jupytext format, or the part of it that the tutorial
needs. This script reads the format itself, without jupytext. The rules:

  - The file may open with a YAML header between "---" lines. Its "jupyter"
    mapping becomes the notebook metadata, without the "jupytext" entry.
  - A block that starts with the line "```python" and ends with the line "```"
    becomes a code cell.
  - All other text becomes markdown cells. Two blank lines in a row start a new
    markdown cell, except inside a fenced block.

Cells get numbered ids, so the same input always gives the same output.
"""

import json
import sys

import yaml


def convert(text):
  """Returns the .ipynb text for the markdown notebook text."""
  lines = text.splitlines()
  metadata, body_start = _split_header(lines)
  cells = []
  for index, (cell_type, source) in enumerate(_parse_cells(lines, body_start)):
    cell = {
      "cell_type": cell_type,
      "id": f"cell-{index}",
      "metadata": {},
      "source": _source(source),
    }
    if cell_type == "code":
      cell["execution_count"] = None
      cell["outputs"] = []
    cells.append(cell)

  notebook = {
    "cells": cells,
    "metadata": metadata,
    "nbformat": 4,
    "nbformat_minor": 5,
  }
  return (
    json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
  )


def _split_header(lines):
  """Returns the notebook metadata and the index of the first body line."""
  if not lines or lines[0] != "---":
    return {}, 0
  end = _closing_line(lines, "---", 0, "YAML header")
  header = yaml.safe_load("\n".join(lines[1:end])) or {}
  metadata = dict(header.get("jupyter", {}))
  metadata.pop("jupytext", None)
  return metadata, end + 1


def _parse_cells(lines, start):
  """Returns (cell_type, source_lines) pairs for lines[start:], in order."""
  cells = []
  markdown = []
  in_fence = False
  index = start
  while index < len(lines):
    line = lines[index]
    if line == "```python" and not in_fence:
      end = _closing_line(lines, "```", index, "```python block")
      cells.extend(("markdown", c) for c in _markdown_cells(markdown))
      cells.append(("code", lines[index + 1 : end]))
      markdown = []
      index = end + 1
      continue
    # Track other fenced blocks, so a "```python" line inside one stays
    # markdown.
    in_fence = _in_fence_after(line, in_fence)
    markdown.append(line)
    index += 1
  cells.extend(("markdown", c) for c in _markdown_cells(markdown))
  return cells


def _closing_line(lines, marker, start, what):
  """Returns the index of the marker line that closes the block at start."""
  try:
    return lines.index(marker, start + 1)
  except ValueError:
    raise ValueError(
      f'{what} opened at line {start + 1} has no closing "{marker}" line'
    ) from None


def _markdown_cells(lines):
  """Splits markdown lines into cell sources at double blank lines."""
  cells = [[]]
  in_fence = False
  blank_run = 0
  for line in lines:
    in_fence = _in_fence_after(line, in_fence)
    if not in_fence and line == "":
      blank_run += 1
      if blank_run == 2:
        cells.append([])
        continue
    else:
      blank_run = 0
    cells[-1].append(line)

  sources = []
  for cell in cells:
    while cell and cell[0] == "":
      cell.pop(0)
    while cell and cell[-1] == "":
      cell.pop()
    if cell:
      sources.append(cell)
  return sources


def _in_fence_after(line, in_fence):
  """Returns whether a markdown fenced block is open after the line."""
  if in_fence:
    return line != "```"
  return line.startswith("```")


def _source(lines):
  """Formats lines as an .ipynb source list, newlines kept but the last."""
  return [line + "\n" for line in lines[:-1]] + lines[-1:]


def main():
  source, destination = sys.argv[1], sys.argv[2]
  with open(source, encoding="utf-8") as f:
    text = f.read()
  with open(destination, "w", encoding="utf-8") as f:
    f.write(convert(text))


if __name__ == "__main__":
  main()
