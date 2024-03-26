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

# Generic Makefile target for ARM Cortex A builds.

FPU := neon
FLOAT_ABI := softfp

GCC_TARGET_ARCH :=
GCC_TARGET_CPU :=

ifeq ($(TARGET_ARCH), armv8.2-a)
  GCC_TARGET_ARCH := armv8.2-a

else ifeq ($(TARGET_ARCH), armv7-a)
  GCC_TARGET_ARCH := armv7-a

else ifeq ($(TARGET_ARCH), cortex-a32)
  GCC_TARGET_CPU := cortex-a32

else
  $(error "TARGET_ARCH=$(TARGET_ARCH) is not supported")
endif

TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

ifneq ($(GCC_TARGET_ARCH),)
  FLAGS_GCC += -march=$(GCC_TARGET_ARCH)
else ifneq ($(GCC_TARGET_CPU),)
  FLAGS_GCC += -mcpu=$(GCC_TARGET_CPU)
endif

CXXFLAGS += $(FLAGS_GCC)
CCFLAGS += $(FLAGS_GCC)

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mfloat-abi=$(FLOAT_ABI) \
  -mfpu=$(FPU) \
  -funsigned-char \
  -mlittle-endian \
  -Wno-type-limits \
  -Wno-unused-private-field \
  -fomit-frame-pointer \
  -MD

ifneq ($(PIC),)
  PLATFORM_FLAGS += -fpic
endif

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

