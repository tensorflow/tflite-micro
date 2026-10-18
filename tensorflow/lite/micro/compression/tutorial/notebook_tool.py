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
"""Opens or runs the tutorial notebook in a Colab-like environment.

    bazel run //tensorflow/lite/micro/compression/tutorial:lab
        Starts JupyterLab with the markdown source open as text. To edit
        and run it as a notebook, right-click it in the file browser and
        choose Open With > Jupytext Notebook. Edits save to the source
        tree; run :ipynb.update afterward. Arguments after -- go to
        jupyter lab, e.g., --ip=<address>.

    bazel run //tensorflow/lite/micro/compression/tutorial:run
        Runs the notebook and writes the result, with outputs, to the
        directory bazel was invoked from.

The notebook runs in a virtual environment, kept in the user cache
directory, that holds the Colab-preinstalled packages the notebook uses,
plus JupyterLab. The notebook installs the released tflite-micro wheel
itself, so it does not test the working tree. The first run downloads
packages and needs network access.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path("tensorflow/lite/micro/compression/tutorial")
NOTEBOOK = "mnist_compression_tutorial.md"
EXECUTED = "mnist_compression_tutorial.executed.ipynb"

# Packages Colab preinstalls that the notebook uses, plus JupyterLab. The
# notebook installs tflite-micro and tensorflow-model-optimization itself.
#
# setuptools provides distutils for tensorflow-model-optimization, and
# before 81, pkg_resources for the released tflite-micro wheel. Colab ships
# setuptools 80.
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


def _runs(python):
  """Returns whether the interpreter at the path starts."""
  try:
    return (
      subprocess.run([python, "--version"], capture_output=True).returncode == 0
    )
  except OSError:
    return False


def ensure_venv():
  """Creates the environment if needed and returns its bin directory."""
  venv = venv_dir()
  python = venv / "bin" / "python"

  # The venv links to bazel's interpreter, which goes away if bazel's cache
  # is cleaned.
  if venv.exists() and not _runs(python):
    print(f"Recreating broken environment {venv}")
    shutil.rmtree(venv)

  if not venv.exists():
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
    # Replace this process so ctrl-C reaches the server.
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
