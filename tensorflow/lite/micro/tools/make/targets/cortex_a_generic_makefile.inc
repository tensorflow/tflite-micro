# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

FLOAT :=hard
GCC_TARGET_ARCH := $(TARGET_ARCH)

# Explicitly set this to true to include the kissfft symbols.
INCLUDE_MICRO_SPEECH := false

ifeq ($(TARGET_ARCH), cortex-a53)
  CORE=A5
  ARM_LDFLAGS := -Wl,--cpu=Cortex-A53 -mfpu=crypto-neon-fp-armv8
  GCC_TARGET_ARCH := cortex-a53
  FLOAT := soft

else ifeq ($(TARGET_ARCH), cortex-a53+fp)
  CORE=A5
  ARM_LDFLAGS := -Wl,--cpu=Cortex-A53 -mfpu=crypto-neon-fp-armv8
  GCC_TARGET_ARCH := cortex-a53
  FLOAT :=hard

else ifeq ($(TARGET_ARCH), cortex-a53+sfp)
  CORE=A5
  ARM_LDFLAGS := -Wl,--cpu=Cortex-A53 -mfpu=crypto-neon-fp-armv8
  GCC_TARGET_ARCH := cortex-a53
  FLOAT=softfp

else ifeq ($(TARGET_ARCH), cortex-a72)
  CORE=A7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-A72 -mfpu=crypto-neon-fp-armv8
  GCC_TARGET_ARCH := cortex-a72
  FLOAT := soft


else ifeq ($(TARGET_ARCH), cortex-a72+fp)
  CORE=A7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-A72 -mfpu=crypto-neon-fp-armv8
  GCC_TARGET_ARCH := cortex-a72
  FLOAT :=hard

else ifeq ($(TARGET_ARCH), cortex-a72+sfp)
  CORE=A7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-A72 -mfpu=crypto-neon-fp-armv8
  GCC_TARGET_ARCH := cortex-a72
  FLOAT=softfp


else ifeq ($(TARGET_ARCH), project_generation)
 # None of the parameters matter since we are not going to build any code.
else
  $(error "TARGET_ARCH=$(TARGET_ARCH) is not supported")
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
  TARGET_DEFAULT_TOOLCHAIN_ROOT := ${TENSORFLOW_ROOT}$(MAKEFILE_DIR)/downloads/gcc_embedded/bin/
  TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
  ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
    ifneq ($(DOWNLOAD_RESULT), SUCCESS)
      $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
    endif
  endif

  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH) -mfpu=auto
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)

else ifeq ($(TOOLCHAIN), arm-none-linux-gnueabihf)
  TARGET_DEFAULT_TOOLCHAIN_ROOT := ${GNU_INSTALL_ROOT}/bin/
  TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)

  TARGET_TOOLCHAIN_PREFIX := arm-none-linux-gnueabihf-

  FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH) -mfpu=auto
  CXXFLAGS += $(FLAGS_GCC)
  CXXFLAGS += -fPIC
  CCFLAGS += $(FLAGS_GCC)
  CCFLAGS += -fPIC

else
  $(error "TOOLCHAIN=$(TOOLCHAIN) is not supported.")
endif

# TODO(#47718): resolve warnings.
OMIT_ERRORS = \
  -Wno-implicit-fallthrough \
  -Wno-unused-variable

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mthumb \
  -mfloat-abi=$(FLOAT) \
  -funsigned-char \
  -mlittle-endian \
   ${OMIT_ERRORS} \
  -Wno-type-limits \
  -Wno-unused-private-field \
  -fomit-frame-pointer \
  -MD \
  -DCPU_$(CORE)=1 \
  $(TARGET_SPECIFIC_FLAGS)

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)


# We only include micro_speech for project generation to allow for all the files
# to be downloaded. We do not include it for an actual build with the
# cortex_m_generic target to prevent kissfft symbols from getting included in
# libtensorflow-microlite.a which can result in symbol collision.
ifneq ($(TARGET_ARCH), project_generation)
  EXCLUDED_EXAMPLE_TESTS := \
    tensorflow/lite/micro/examples/micro_speech/Makefile.inc
  MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))
endif
