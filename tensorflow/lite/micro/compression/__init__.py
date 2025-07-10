# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite for Microcontrollers compression module."""

# This __init__.py file exists to make compression features available as part
# of the tflite_micro Python package.
#
# Usage example:
#   from tflite_micro import compression
#   ...

from .compress import compress
from .spec import parse_yaml
from .spec_builder import SpecBuilder

__all__ = ["compress", "parse_yaml", "SpecBuilder"]
