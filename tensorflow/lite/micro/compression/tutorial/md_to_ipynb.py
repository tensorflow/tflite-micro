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
"""Convert a jupytext markdown notebook to .ipynb, deterministically.

jupytext assigns random cell ids when writing .ipynb, which would make the
generated notebook differ on every conversion. Assign sequential ids instead,
so the output is a pure function of the input and can be compared against a
copy saved in the source tree. See the bazel target
":ipynb".
"""

import sys

import jupytext


def main():
  source, destination = sys.argv[1], sys.argv[2]
  notebook = jupytext.read(source, fmt="md")
  for index, cell in enumerate(notebook.cells):
    cell["id"] = f"cell-{index}"
  jupytext.write(notebook, destination, fmt="ipynb")


if __name__ == "__main__":
  main()
