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
"""Execute a markdown notebook headlessly: execute.py SOURCE OUTPUT WORKDIR.

Runs inside the notebook environment created by notebook_tool.py, which
provides jupytext, nbclient, and the kernel. Reads the jupytext markdown
notebook SOURCE, executes it with the kernel working directory WORKDIR, and
writes the executed notebook, with outputs, to OUTPUT.
"""

import os
import sys

import jupytext
import nbclient

CELL_TIMEOUT_SECONDS = 1800


def main():
  source, output, workdir = sys.argv[1:4]

  # Render plots off-screen.
  os.environ.setdefault("MPLBACKEND", "Agg")

  notebook = jupytext.read(source, fmt="md")
  client = nbclient.NotebookClient(
    notebook,
    timeout=CELL_TIMEOUT_SECONDS,
    kernel_name="python3",
    resources={"metadata": {"path": workdir}},
  )
  client.execute()

  jupytext.write(notebook, output, fmt="ipynb")
  print(f"Executed notebook written to {output}")


if __name__ == "__main__":
  main()
