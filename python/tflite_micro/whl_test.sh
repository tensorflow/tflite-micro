#!/usr/bin/sh

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

# Install the given tflm-micro.whl in a fresh virtual environment and run its
# embedded, post-installation checks.

set -e

WHL="${1}"

# Create venv for this test.
python3 -m venv pyenv
. pyenv/bin/activate

# Disable pip's cache for two reasons: 1) the default location in
# $XDG_CACHE_HOME causes errors when pip is run from a bazel sandbox, and 2) it
# makes no sense to relocate the cache within the sandbox since files generated
# in the sandbox are deleted after the run.
export PIP_NO_CACHE_DIR=true

# Test package installation.
pip install "${WHL}"
pip show --files tflite-micro

# Run the package's post-installation checks.
python3 << HEREDOC
import sys, tflite_micro
print(tflite_micro.__version__)
sys.exit(0 if tflite_micro.postinstall_check.passed() else 1)
HEREDOC
