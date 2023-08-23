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
# ----

# Define a public API for the package by providing aliases for modules which
# are otherwise deeply nested in subpackages determined by their location in
# the tflm source tree. Directly using modules and subpackages not explicitly
# made part of the public API in code outside of the tflm source tree is
# unsupported.

from tflite_micro.python.tflite_micro import runtime

# Unambiguously identify the source used to build the package.
from tflite_micro.python.tflite_micro._version import __version__

# Ordered after `runtime` to avoid a circular dependency
from tflite_micro.python.tflite_micro import postinstall_check
