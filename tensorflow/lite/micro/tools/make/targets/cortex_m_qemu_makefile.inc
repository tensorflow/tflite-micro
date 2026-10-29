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

ifeq ($(TARGET_TOOLCHAIN_ROOT),)
  ifeq ($(shell which arm-none-eabi-gcc 2>/dev/null),)
    TARGET_DEFAULT_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/gcc_embedded/bin/
    TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${DOWNLOADS_DIR} $(TENSORFLOW_ROOT))
    ifneq ($(DOWNLOAD_RESULT), SUCCESS)
      $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
    endif
  endif
endif
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mcpu=$(TARGET_ARCH) \
  -mthumb \
  -mfloat-abi=soft \
  -mfpu=auto \
  -funsigned-char \
  -mlittle-endian \
  -fomit-frame-pointer

# Enable semihosting for QEMU to enable host system calls for debug_log and micro_time.
PLATFORM_FLAGS += --specs=rdimon.specs

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

# TODO(b/158651472): Fix the memory_arena_threshold_test
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_qemu.sh arm $(TARGET_ARCH)
