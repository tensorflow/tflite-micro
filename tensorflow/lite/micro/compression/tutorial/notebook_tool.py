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
"""Serve or execute the tutorial notebook in a Colab-like environment.

Two bazel targets share this tool:

    bazel run //tensorflow/lite/micro/compression/tutorial:lab
        Opens the notebook's markdown source in JupyterLab for interactive
        editing and execution. Edits save straight to the source tree;
        after editing, regenerate the committed .ipynb by running the
        bazel target :ipynb.update. Arguments after -- pass through to
        jupyter lab; e.g., to serve beyond localhost, such as to browse
        from another machine on a private network, bind a specific
        address with `bazel run ...:lab -- --ip=<address>`.

    bazel run //tensorflow/lite/micro/compression/tutorial:run
        Executes the notebook end to end, headlessly, and writes the executed
        notebook, with outputs, plus the .tflite files the tutorial saves, to
        the directory from which bazel was invoked.

Rather than running the notebook in the bazel-provided Python environment,
which would drag the tutorial's dependencies into the repository's pinned
requirements, this tool maintains a virtual environment mirroring what Google
Colab preinstalls, and the notebook itself pip-installs the rest, exactly as
it does on Colab. The notebook therefore runs against the released
tflite-micro wheel, not the working tree. The environment is kept under the
user cache directory and reused; delete it to start fresh. Creating it
downloads packages from PyPI, so the first run needs network access and a few
minutes.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path("tensorflow/lite/micro/compression/tutorial")
NOTEBOOK = "mnist_compression_tutorial.md"
EXECUTED = "mnist_compression_tutorial.executed.ipynb"

# What Google Colab preinstalls, as far as the notebook is concerned, plus
# the Jupyter tooling to serve and execute it. The packages the notebook
# installs itself (tflite-micro, tensorflow-model-optimization) are
# deliberately absent.
#
# setuptools shims distutils, which tensorflow-model-optimization imports
# and Python 3.12 removed. Versions before 81 also provide pkg_resources,
# which the currently released tflite-micro wheel still imports; the bound
# can be dropped once a wheel without that import is released.
BASELINE = [
  "ipykernel",
  "jupyterlab",
  "jupytext",
  "matplotlib",
  "nbclient",
  "setuptools<81",
  "tensorflow",
  "tf-keras",
]


def venv_dir():
  cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
  return cache / "tflite-micro" / "mnist-tutorial-venv"


def ensure_venv():
  """Create or reuse the notebook environment; return its bin directory."""
  venv = venv_dir()
  python = venv / "bin" / "python"

  # The venv symlinks the interpreter it was created from, so it dangles if
  # bazel's cache is cleaned. Recreate it if it no longer runs.
  if python.exists():
    if subprocess.run([python, "--version"], capture_output=True).returncode:
      print(f"Recreating broken environment {venv}")
      shutil.rmtree(venv)

  if not python.exists():
    print(f"Creating notebook environment {venv}")
    print("The first run downloads packages from PyPI and takes a few minutes.")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

  subprocess.run(
    [venv / "bin" / "pip", "install", "--quiet", *BASELINE], check=True
  )
  return venv / "bin"


def main():
  mode = sys.argv[1]
  extra = sys.argv[2:]

  workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if not workspace:
    sys.exit("Run this tool via `bazel run`; see the module docstring.")
  source = Path(workspace) / PACKAGE / NOTEBOOK

  bindir = ensure_venv()

  if mode == "lab":
    # Serve the source tree's file so edits land there, and replace this
    # process so ctrl-C reaches the server directly.
    os.execv(bindir / "jupyter", ["jupyter", "lab", *extra, str(source)])
  elif mode == "execute":
    workdir = os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd())
    helper = PACKAGE / "notebook_execute.py"
    result = subprocess.run(
      [
        bindir / "python",
        helper,
        source,
        Path(workdir) / EXECUTED,
        workdir,
      ]
    )
    sys.exit(result.returncode)
  else:
    sys.exit(f"Unknown mode {mode!r}; expected 'lab' or 'execute'.")


if __name__ == "__main__":
  main()
