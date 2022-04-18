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

PYTHON_EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)

SRCS += \
	tensorflow/lite/micro/tools/pybind_wrapper/interpreter_wrapper.cc \
	tensorflow/lite/micro/tools/pybind_wrapper/interpreter_wrapper_pybind.cc \
	tensorflow/lite/micro/tools/pybind_wrapper/python_error_reporter.cc \
	tensorflow/lite/micro/tools/pybind_wrapper/python_utils.cc \
	tensorflow/lite/micro/tools/pybind_wrapper/numpy.cc \

# TODO: -I$(MAKEFILE_DIR)/downloads/flatbuffers/include
INCLUDES += \
	$(PYBIND11_INCLUDES) \
	-I. \
	-Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include \

# Must compile with -fPIC
# make -f tensorflow/lite/micro/tools/make/Makefile CXXFLAGS=-fPIC build
LDFLAGS = -Ltensorflow/lite/micro/tools/make/gen/linux_x86_64_default/lib/ -ltensorflow-microlite
# LDFLAGS = --whole-archive -shared -Wl tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/lib/libtensorflow-microlite.a

# Changed to g++ because gcc results in ImportError: /home/psho/tflite-micro/tflm_interpreter.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZTVN10__cxxabiv117__class_type_infoE
CC = g++

CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC

tflm_interpreter$(PYTHON_EXTENSION_SUFFIX):
	$(CC) $(CFLAGS) $(SRCS) $(INCLUDES) $(LDFLAGS) -o tflm_interpreter$(PYTHON_EXTENSION_SUFFIX)

clean:
	rm -rf tflm_interpreter$(PYTHON_EXTENSION_SUFFIX)
