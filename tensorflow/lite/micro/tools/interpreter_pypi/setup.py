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

import sys

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension
import setuptools
from glob import glob

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

ext_modules = [
    Pybind11Extension(
        "interpreter_wrapper_pybind",
        sorted(
            glob(
                "/home/psho/tflite-micro/tensorflow/lite/micro/tools/pybind_wrapper/*.cc"
            )),  # Sort source files for reproducibility
        include_dirs=[
            "/home/psho/tflite-micro",
            "/home/psho/tflite-micro/tensorflow/lite/micro/tools/pybind_wrapper",
            "/home/psho/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include"
        ],
        library_dirs=[
            "/home/psho/tflite-micro/tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/lib/"
        ],
        libraries=["tensorflow-microlite"],
    ),
]

setuptools.setup(
    name="example-tflm-interpreter-psho",
    version="0.0.3",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    ext_modules=ext_modules,
)
