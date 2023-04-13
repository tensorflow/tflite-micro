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
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh  ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mcpu=$(TARGET_ARCH) \
  -mthumb \
  -mfloat-abi=soft \
  -mfpu=auto \
  -funsigned-char \
  -mlittle-endian \
  -fomit-frame-pointer \

# Enable semihosting for QEMU to enable host system calls for debug_log and micro_time.
PLATFORM_FLAGS += --specs=rdimon.specs
# USE_SEMIHOSTING enables host system calls for debug_log and micro_time.
# Please see micro/cortex_m_generic micro_time.cc and debug_log.cc for more.
PLATFORM_FLAGS += -DUSE_SEMIHOSTING

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

# Excludes micro_allocator_test because it calls CreateQuantizedFlatbufferTensor,
# which use std::vector constructor which then invokes new.
# Excludes memory_arena_threshold_test because the size difference of some
# allocator classes between different architectures.
# TODO(b/158651472): Fix the memory_arena_threshold_test
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_allocator_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

# flatbuffer_utils_test is intentionaly disabled because the flexbuffer builder
# uses dynamic memory.
EXCLUDED_TESTS += $(TENSORFLOW_ROOT)tensorflow/lite/micro/flatbuffer_utils_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_qemu.sh arm $(TARGET_ARCH)
