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
# ==============================================================================

export PATH := $(DOWNLOADS_DIR)/gcc_embedded/bin/:$(PATH)
TARGET_ARCH := cortex-m3
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/renode_download.sh  ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the renode download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh  ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/stm32_bare_lib_download.sh  ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the STM32 Bare Lib download: $(DOWNLOAD_RESULT))
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mcpu=cortex-m3 \
  -mthumb \
  -Wno-vla \
  -Wno-shadow \
  -fomit-frame-pointer \
  -nostdlib

# TODO(b/168334217): Currently we always add -DNDEBUG because the build is
# broken w/o it. Remove this workaround once the issue is resolved.
PLATFORM_FLAGS += -DNDEBUG

# TODO(#46937): Remove once initialization of global variables is sorted out.
PLATFORM_FLAGS += -DRENODE

CXXFLAGS += $(PLATFORM_FLAGS) -fno-use-cxa-atexit
CCFLAGS += $(PLATFORM_FLAGS)

LDFLAGS += \
  -T $(MAKEFILE_DIR)/targets/bluepill/bluepill.lds \
  -Wl,--no-warn-rwx-segment \
  -Wl,-Map=gen/$(TARGET).map,--cref

# Additional include paths needed for the stm_32_bare_lib only.
INCLUDES += \
  -isystem$(DOWNLOADS_DIR)/cmsis/CMSIS/Core/Include/ \
  -I$(DOWNLOADS_DIR)/stm32_bare_lib/include

MICROLITE_CC_SRCS += \
  $(wildcard $(DOWNLOADS_DIR)/stm32_bare_lib/source/*.c) \
  $(wildcard $(DOWNLOADS_DIR)/stm32_bare_lib/source/*.cc)
EXCLUDED_SRCS := \
  $(DOWNLOADS_DIR)/stm32_bare_lib/source/debug_log.c
MICROLITE_CC_SRCS := $(filter-out $(EXCLUDED_SRCS), $(MICROLITE_CC_SRCS))

# Excludes micro_allocator_test because it calls CreateQuantizedFlatbufferTensor,
# which use std::vector constructor which then invokes new.
# Excludes memory_arena_threshold_test because the size difference of some
# allocator classes between different architectures.
# Excludes hexdump_test because it fails in release model when MicroVsnprintf
# isn't defined.
# TODO(b/158651472): Fix the memory_arena_threshold_test
# TODO(b/b/410801324): Fix hexdump_test for Bluepill
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_allocator_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/hexdump_test.cc

# flatbuffer_utils_test is intentionaly disabled because the flexbuffer builder
# uses dynamic memory.
EXCLUDED_TESTS += $(TENSORFLOW_ROOT)tensorflow/lite/micro/flatbuffer_utils_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

EXCLUDED_EXAMPLE_TESTS :=

MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_renode.sh

# We are setting this variable to non-zero to allow us to have a custom
# implementation of `make test` for bluepill
TARGET_SPECIFIC_MAKE_TEST := 1

TEST_TARGET_BINARIES = $(shell ls -1 $(BINDIR)/*_test)

test: build
	$(TEST_SCRIPT) "$(TEST_TARGET_BINARIES)" $(TEST_PASS_STRING) $(TARGET)

include $(MAKEFILE_DIR)/ext_libs/eyalroz_printf.inc
