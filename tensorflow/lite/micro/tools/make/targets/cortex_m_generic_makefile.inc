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

# Generic Makefile target for ARM Cortex M builds.
# For more info see: tensorflow/lite/micro/cortex_m_generic/README.md

# Needed in case running without OPTIMIZED_KERNEL_DIR=cmsis_nn.
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
FPU :=
GCC_TARGET_ARCH := $(TARGET_ARCH)
SIGNED_CHAR := false

# Explicitly set this to true to include the kissfft symbols.
INCLUDE_MICRO_SPEECH := false

ifeq ($(TARGET_ARCH), cortex-m0)
  CORE=M0
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M0

else ifeq ($(TARGET_ARCH), cortex-m0plus)
  CORE=M0plus
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M0plus

else ifeq ($(TARGET_ARCH), cortex-m3)
  CORE=M3
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M3

else ifeq ($(TARGET_ARCH), cortex-m33)
  CORE=M33
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M33
  FLOAT=hard

else ifeq ($(TARGET_ARCH), cortex-m33+nodsp)
  CORE=M33
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M33.no_dsp.no_fp

else ifeq ($(TARGET_ARCH), cortex-m4)
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4.no_fp
  GCC_TARGET_ARCH := cortex-m4+nofp

else ifeq ($(TARGET_ARCH), cortex-m4+fp)
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4
  FLOAT=hard
  GCC_TARGET_ARCH := cortex-m4

else ifeq ($(TARGET_ARCH), cortex-m4+sfp)
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4
  FLOAT=softfp
  GCC_TARGET_ARCH := cortex-m4

else ifeq ($(TARGET_ARCH), cortex-m55)
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.fp
  FLOAT=hard

else ifeq ($(TARGET_ARCH), cortex-m55+nodsp+nofp)
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_dsp.no_fp

else ifeq ($(TARGET_ARCH), cortex-m55+nofp)
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_fp

else ifeq ($(TARGET_ARCH), cortex-m7)
  CORE=M7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M7.no_fp
  GCC_TARGET_ARCH := cortex-m7+nofp

else ifeq ($(TARGET_ARCH), cortex-m7+fp)
  CORE=M7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M7
  FLOAT=hard
  GCC_TARGET_ARCH := cortex-m7

else ifeq ($(TARGET_ARCH), cortex-m85)
  CORE=M85
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.fp
  FLOAT=hard

else ifeq ($(TARGET_ARCH), project_generation)
  # No flags needed here as project_generation does not build anything.
else
  $(error "TARGET_ARCH=$(TARGET_ARCH) is not supported")
endif

# Dependency to CMSIS-Device for DWT/PMU counters.
ARM_CPU := "ARMC$(CORE)"
INCLUDES += \
  -I$(CORTEX_DFP_PATH)/Device/$(ARM_CPU)/Include \
  -I$(CMSIS_PATH)/CMSIS/Core/Include

ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
  # soft-abi=soft disables MVE - use softfp instead for M55.
  ifeq ($(FLOAT),soft)
    FLOAT=softfp
  endif
endif

# Toolchain specfic flags
ifeq ($(TOOLCHAIN), armclang)
  CXX_TOOL  := armclang
  CC_TOOL   := armclang
  AR_TOOL   := armar
  LD        := armlink

  FLAGS_ARMC = \
    --target=arm-arm-none-eabi \
    -mcpu=$(TARGET_ARCH)

  CXXFLAGS += $(FLAGS_ARMC)
  CCFLAGS += $(FLAGS_ARMC)
  LDFLAGS += $(ARM_LDFLAGS)

  # Arm Compiler will not link the Math library (see below), therefore we're filtering it out.
  # See Fatal error: L6450U: Cannot find library m:
  # "Arm Compiler is designed to run in a bare metal environment,
  # and automatically includes implementations of these functions,
  # and so no such flag is necessary."
  # https://developer.arm.com/documentation/100891/0611/troubleshooting/general-troubleshooting-advice
  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

else ifeq ($(TOOLCHAIN), gcc)
  TARGET_DEFAULT_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/gcc_embedded/bin/
  TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
  ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${DOWNLOADS_DIR} ${TENSORFLOW_ROOT})
    ifneq ($(DOWNLOAD_RESULT), SUCCESS)
      $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
    endif
  endif

  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH)
  ifeq ($(TARGET_ARCH), cortex-m4)
    FLAGS_GCC += -mfpu=fpv4-sp-d16
  else ifeq ($(TARGET_ARCH), cortex-m7)
    FLAGS_GCC += -mfpu=fpv4-sp-d16
  else ifneq ($(FPU),)
    FLAGS_GCC += -mfpu=$(FPU)
  else
    FLAGS_GCC += -mfpu=auto
  endif
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)

else
  $(error "TOOLCHAIN=$(TOOLCHAIN) is not supported.")
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mthumb \
  -mfloat-abi=$(FLOAT) \
  -mlittle-endian \
  -Wno-type-limits \
  -Wno-unused-private-field \
  -fomit-frame-pointer \
  -MD \
  -DCPU_$(CORE)=1

ifeq ($(SIGNED_CHAR), false)
  PLATFORM_FLAGS += -funsigned-char
else
  PLATFORM_FLAGS += -fsigned-char
endif

# For DWT/PMU counters. Header file name is depending on target architecture.
PLATFORM_FLAGS += -DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE=\"$(ARM_CPU).h\"
PLATFORM_FLAGS += -D$(ARM_CPU)

# Arm Cortex-M55 and Cortex-M85 use PMU counters.
ifneq ($(filter "ARMCM55" "ARMCM85",$(ARM_CPU)),)
  PLATFORM_FLAGS += -DARM_MODEL_USE_PMU_COUNTERS
endif

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

# Needed for the project generation interface.
MICROLITE_CC_HDRS += \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h

# We only include micro_speech for project generation to allow for all the files
# to be downloaded. We do not include it for an actual build with the
# cortex_m_generic target to prevent kissfft symbols from getting included in
# libtensorflow-microlite.a which can result in symbol collision.
ifneq ($(TARGET_ARCH), project_generation)
  EXCLUDED_EXAMPLE_TESTS := \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/Makefile.inc
  MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))
endif

