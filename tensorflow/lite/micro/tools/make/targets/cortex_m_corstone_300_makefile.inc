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
CMSIS_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
CORTEX_DFP_PATH := $(CMSIS_PATH)/Cortex_DFP
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
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
    -mcpu=$(MCPU_OPTION)

  # Pass comma separated linker options to armlink
  ARMC6_LDFLAGS += -Wl,--strict,--summary_stderr,--info,summarysizes,--map
  ARMC6_LDFLAGS += -Wl,--load_addr_map_info,--xref,--callgraph,--symbols
  ARMC6_LDFLAGS += -Wl,--info,sizes,--info,totals,--info,unused,--info,veneers
  ARMC6_LDFLAGS += -Wl,--list=gen/$(TARGET).map
  ARMC6_LDFLAGS += -Wl,--entry=Reset_Handler --verbose
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

  # This does not build with armclang and is anyway not used by this target.
  EXCLUDED_TESTS := \
    tensorflow/lite/micro/tools/benchmarking/Makefile.inc
  MICRO_LITE_BENCHMARKS := $(filter-out $(EXCLUDED_TESTS), $(MICRO_LITE_BENCHMARKS))

else ifeq ($(TOOLCHAIN), gcc)
  TARGET_DEFAULT_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/gcc_embedded/bin/
  TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
  ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
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
