# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Settings for not pre-defined ARC processors. 
# User need to specify ARC target with Tool Configuration File (*.tcf). 
# Path to this file must be passed through TCF_FILE variable.
# Otherwise, default em7d_voice_audio configuration is used

TARGET_ARCH := arc
ARC_TOOLCHAIN := mwdt

# Overriding TARGET variable to change name of project folder according
# to specified Tool Configuration File (*.tcf) passed through TCF_FILE variable
# or default em7d_voice_audio configuration.
ifneq ($(TCF_FILE),)
  override TARGET = $(basename $(notdir $(TCF_FILE)))
else
  $(warning TCF_FILE variable is not specified. Use default em7d_voice_audio configuration)
  override TARGET = em7d_voice_audio
  TCF_FILE = em7d_voice_audio
endif

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  override TARGET := $(TARGET)_mli20
endif

include $(MAKEFILE_DIR)/targets/arc/arc_common.inc

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  CXXFLAGS += -DMLI_2_0
  CCFLAGS += -DMLI_2_0

# If kernel tests running - using define to activate online permutation.
ifneq ($(findstring test, $(MAKECMDGOALS)),)
  CXXFLAGS += -DMLI_2_0_KRNL_TEST
  CCFLAGS += -DMLI_2_0_KRNL_TEST
endif

ifneq ($(BUILD_LIB_DIR), )
  LDFLAGS += -Hlib=$(BUILD_LIB_DIR)
else
  $(warning BUILD_LIB_DIR variable is not specified. Default will be used.)
endif

endif # ARC_TAGS

EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

EXCLUDED_EXAMPLE_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/Makefile.inc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/memory_footprint/Makefile.inc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/Makefile.inc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/network_tester/Makefile.inc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/Makefile.inc

MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

# Removing integration tests for ARC due to issues.
MICRO_LITE_INTEGRATION_TESTS := 

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_arc_mdb.sh

TARGET_SPECIFIC_MAKE_TEST := 1

TEST_TARGET_BINARIES = $(shell ls -1 $(BINDIR)*_test)

# Removing conv test due to bug in one of cases.
EXCLUDED_TEST_BINARIES := \
  $(BINDIR)kernel_conv_test

TEST_TARGET_BINARIES := $(filter-out $(EXCLUDED_TEST_BINARIES), $(TEST_TARGET_BINARIES))

test: build
	$(foreach test,$(TEST_TARGET_BINARIES),$(TEST_SCRIPT) $(test) $(TCF_FILE) $(TEST_PASS_STRING) || exit;)
