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
# ==============================================================================

# ARM Cortex M makefile targeted for a FVP based on Arm Corstone-300 software.
# For more info see: tensorflow/lite/micro/cortex_m_corstone_300/README.md

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M), aarch64)
  export PATH := $(DOWNLOADS_DIR)/corstone300/models/Linux64_armv8l_GCC-9.3:$(PATH)
else
  export PATH := $(DOWNLOADS_DIR)/corstone300/models/Linux64_GCC-9.3:$(PATH)
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/corstone_300_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the Arm Corstone-300 software download: $(DOWNLOAD_RESULT))
endif

ETHOS_U_CORE_PLATFORM := $(DOWNLOADS_DIR)/ethos_u_core_platform/targets/corstone-300
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ethos_u_core_platform_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the Ethos-U Core Platform software download: $(DOWNLOAD_RESULT))
endif

# This target has dependencies to CMSIS-Device so just in case running without OPTIMIZED_KERNEL_DIR=cmsis_nn.
include $(MAKEFILE_DIR)/ext_libs/cmsis.inc

FLOAT := soft
MCPU_OPTION := $(TARGET_ARCH)

ifeq ($(TARGET_ARCH), cortex-m0)
  # Default FLOAT=soft, MCPU_OPTION=cortex-m0
else ifeq ($(TARGET_ARCH), cortex-m3)
  # Default FLOAT=soft, MCPU_OPTION=cortex-m3
else ifeq ($(TARGET_ARCH), cortex-m4)
  MCPU_OPTION := cortex-m4+nofp
else ifeq ($(TARGET_ARCH), cortex-m4+fp)
  FLOAT=hard
  MCPU_OPTION := cortex-m4
else ifeq ($(TARGET_ARCH), cortex-m55)
  FLOAT=hard
else ifeq ($(TARGET_ARCH), cortex-m7)
  MCPU_OPTION := cortex-m7+nofp
else ifeq ($(TARGET_ARCH), cortex-m7+fp)
  FLOAT=hard
  MCPU_OPTION := cortex-m7
else
  $(error "TARGET_ARCH=$(TARGET_ARCH) is not supported")
endif

ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
  # soft-abi=soft disables MVE - use softfp instead for M55.
  ifeq ($(FLOAT),soft)
    FLOAT=softfp
  endif
endif

# Filter out part of mcpu string for choosing the correct startup files
ARM_CPU := $(subst cortex-m,ARMCM,$(MCPU_OPTION))
ARM_CPU := $(subst +nofp,,$(ARM_CPU))

ifeq ($(TOOLCHAIN), gcc)
  ifeq ($(TARGET_TOOLCHAIN_ROOT),)
    ifeq ($(shell which arm-none-eabi-gcc 2>/dev/null),)
      TARGET_DEFAULT_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/gcc_embedded/bin/
      TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
      DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
      ifneq ($(DOWNLOAD_RESULT), SUCCESS)
        $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
      endif
    endif
  endif
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  FLAGS_GCC = -mcpu=$(MCPU_OPTION) -mfpu=auto
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)

  LDFLAGS += \
    --specs=nosys.specs \
    -T $(ETHOS_U_CORE_PLATFORM)/platform_parsed.ld \
    -Wl,-Map=gen/$(TARGET).map,--cref \
    -Wl,--gc-sections \
    --entry Reset_Handler

  ldflags_to_remove = -Wl,--fatal-warnings
  LDFLAGS := $(filter-out $(ldflags_to_remove),$(LDFLAGS))

else
  $(error "TOOLCHAIN=$(TOOLCHAIN) is not supported.")
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mthumb \
  -mfloat-abi=$(FLOAT) \
  -funsigned-char \
  -mlittle-endian \
  -fomit-frame-pointer \
  -MD

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

CXXFLAGS += -D$(ARM_CPU)
CCFLAGS += -D$(ARM_CPU)

# For Ethos-U Core Driver. Header file name is depending on target architecture.
CXXFLAGS += -DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE=\"$(ARM_CPU).h\"

THIRD_PARTY_CC_SRCS += \
  $(ETHOS_U_CORE_PLATFORM)/retarget.c \
  $(ETHOS_U_CORE_PLATFORM)/uart.c

ifeq ($(CO_PROCESSOR), ethos_u)
  ETHOSU_ARCH=u55
endif

THIRD_PARTY_CC_SRCS += \
  $(CORTEX_DFP_PATH)/Device/$(ARM_CPU)/Source/system_$(ARM_CPU).c \
  $(CORTEX_DFP_PATH)/Device/$(ARM_CPU)/Source/startup_$(ARM_CPU).c
INCLUDES += \
  -I$(CORTEX_DFP_PATH)/Device/$(ARM_CPU)/Include \
  -I$(CMSIS_PATH)/CMSIS/Core/Include

# TODO(#274): Examine why some tests fail here.
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc  \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/recording_micro_allocator_test.cc
ifeq ($(CO_PROCESSOR), ethos_u)
# This does not work with Ethos-U enabled since then NPU PMU counters are used instead for the sake of the benchmark example.
EXCLUDED_TESTS += \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_time_test.cc
endif
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_arm_corstone_300.sh
