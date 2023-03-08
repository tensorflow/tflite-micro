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

# ARM Cortex M makefile targeted for a FVP based on Arm Corstone-300 software.
# For more info see: tensorflow/lite/micro/cortex_m_corstone_300/README.md

export PATH := $(MAKEFILE_DIR)/downloads/corstone300/models/Linux64_GCC-6.4:$(PATH)
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/corstone_300_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the Arm Corstone-300 software download: $(DOWNLOAD_RESULT))
endif

ETHOS_U_CORE_PLATFORM := ${PWD}/$(MAKEFILE_DIR)/downloads/ethos_u_core_platform/targets/corstone-300
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ethos_u_core_platform_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the Ethos-U Core Platform software download: $(DOWNLOAD_RESULT))
endif

# This target has dependencies to CMSIS-Device so just in case running without OPTIMIZED_KERNEL_DIR=cmsis_nn.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif

FLOAT := soft
MCPU_OPTION := $(TARGET_ARCH)

# Linker and targets must match according to:
# https://www.keil.com/support/man/docs/armclang_mig/armclang_mig_aya1488905345341.htm
ifeq ($(TARGET_ARCH), cortex-m0)
  ARMC6_LDFLAGS += -Wl,--cpu=Cortex-M0

else ifeq ($(TARGET_ARCH), cortex-m3)
  ARMC6_LDFLAGS += -Wl,--cpu=Cortex-M3

else ifeq ($(TARGET_ARCH), cortex-m4)
  ARMC6_LDFLAGS += -Wl,--cpu=Cortex-M4.no_fp
  MCPU_OPTION := cortex-m4+nofp

else ifeq ($(TARGET_ARCH), cortex-m4+fp)
  ARMC6_LDFLAGS += -Wl,--cpu=Cortex-M4
  FLOAT=hard
  MCPU_OPTION := cortex-m4
  CMSIS_ARM_FEATURES := _FP

else ifeq ($(TARGET_ARCH), cortex-m55)
  ARMC6_LDFLAGS += -Wl,--cpu=8.1-M.Main.mve.fp
  FLOAT=hard

else ifeq ($(TARGET_ARCH), cortex-m7)
  ARMC6_LDFLAGS += -Wl,--cpu=Cortex-M7.no_fp
  MCPU_OPTION := cortex-m7+nofp

else ifeq ($(TARGET_ARCH), cortex-m7+fp)
  ARMC6_LDFLAGS += -Wl,--cpu=Cortex-M7
  FLOAT=hard
  MCPU_OPTION := cortex-m7
  CMSIS_ARM_FEATURES := _DP

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

ifeq ($(TOOLCHAIN), armclang)
  CXX_TOOL  := armclang
  CC_TOOL   := armclang
  AR_TOOL   := armar
  LD        := armlink

  FLAGS_ARMC = \
    --target=arm-arm-none-eabi \
    -Wno-unused-private-field \
    -mcpu=$(MCPU_OPTION) \
    -ffp-mode=full

  # Pass comma separated linker options to armlink
  ARMC6_LDFLAGS += -Wl,--strict,--summary_stderr,--info,summarysizes,--map
  ARMC6_LDFLAGS += -Wl,--load_addr_map_info,--xref,--callgraph,--symbols
  ARMC6_LDFLAGS += -Wl,--info,sizes,--info,totals,--info,unused,--info,veneers
  ARMC6_LDFLAGS += -Wl,--list=${TENSORFLOW_ROOT}gen/$(TARGET).map
  ARMC6_LDFLAGS += -Wl,--entry=Reset_Handler  --verbose
  ARMC6_LDFLAGS += -Wl,--scatter=$(ETHOS_U_CORE_PLATFORM)/platform.scatter

  # Pass a hint to the linker where to find the entry point. This needs to be
  # done since the startup object file (containing the entry point) is inside
  # the TFLM library. See:
  # https://developer.arm.com/documentation/ka003125/latest
  ARMC6_LDFLAGS += -Wl,$(LIBDIR)/$(MICROLITE_LIB_NAME)\(startup_$(ARM_CPU).o\)

  CXXFLAGS += $(FLAGS_ARMC)
  CCFLAGS += $(FLAGS_ARMC)
  LDFLAGS := $(ARMC6_LDFLAGS)

  MICROLITE_CC_KERNEL_SRCS := $(filter-out $(EXCLUDED_CC_SRCS),$(MICROLITE_CC_KERNEL_SRCS))
  THIRD_PARTY_CC_HDRS := $(filter-out $(EXCLUDED_HDRS),$(THIRD_PARTY_CC_HDRS))
  MICROLITE_CC_HDRS := $(filter-out $(EXCLUDED_KERNEL_HDRS),$(MICROLITE_CC_HDRS))

  # Arm Compiler will not link the Math library (see below), therefore we're filtering it out.
  # See Fatal error: L6450U: Cannot find library m:
  # "Arm Compiler is designed to run in a bare metal environment,
  # and automatically includes implementations of these functions,
  # and so no such flag is necessary."
  # https://developer.arm.com/documentation/100891/0611/troubleshooting/general-troubleshooting-advice
  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

else ifeq ($(TOOLCHAIN), gcc)
  TARGET_DEFAULT_TOOLCHAIN_ROOT := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/
  TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
  ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
    ifneq ($(DOWNLOAD_RESULT), SUCCESS)
      $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
    endif
  endif
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  FLAGS_GCC = -mcpu=$(MCPU_OPTION) -mfpu=auto
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)

  LDFLAGS += \
    --specs=nosys.specs \
    -T $(ETHOS_U_CORE_PLATFORM)/platform_parsed.ld \
    -Wl,-Map=${TENSORFLOW_ROOT}gen/$(TARGET).map,--cref \
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

CXXFLAGS += -D$(ARM_CPU)$(CMSIS_ARM_FEATURES)
CCFLAGS += -D$(ARM_CPU)$(CMSIS_ARM_FEATURES)

# For Ethos-U Core Driver. Header file name is depending on target architecture.
CXXFLAGS += -DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE=\"$(ARM_CPU)$(CMSIS_ARM_FEATURES).h\"

THIRD_PARTY_CC_SRCS += \
  $(ETHOS_U_CORE_PLATFORM)/retarget.c \
  $(ETHOS_U_CORE_PLATFORM)/uart.c

ifeq ($(CO_PROCESSOR), ethos_u)
  ETHOSU_ARCH=u55
endif

CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
THIRD_PARTY_CC_SRCS += \
  $(CMSIS_PATH)/Device/ARM/$(ARM_CPU)/Source/system_$(ARM_CPU).c \
  $(CMSIS_PATH)/Device/ARM/$(ARM_CPU)/Source/startup_$(ARM_CPU).c
INCLUDES += \
  -I$(CMSIS_PATH)/Device/ARM/$(ARM_CPU)/Include \
  -I$(CMSIS_PATH)/CMSIS/Core/Include

# TODO(#274): Examine why some tests fail here.
EXCLUDED_TESTS := \
  tensorflow/lite/micro/memory_arena_threshold_test.cc  \
  tensorflow/lite/micro/recording_micro_allocator_test.cc
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))
EXCLUDED_EXAMPLE_TESTS := \
 tensorflow/lite/micro/examples/hello_world/Makefile.inc
MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

TEST_SCRIPT := tensorflow/lite/micro/testing/test_with_arm_corstone_300.sh
