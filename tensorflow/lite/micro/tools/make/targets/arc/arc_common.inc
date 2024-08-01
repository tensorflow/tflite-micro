# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Common Settings for ARC platform and its projects. 
# Might be reused across different targets

ifeq ($(TARGET_ARCH), arc)

ifeq ($(ARC_TOOLCHAIN), mwdt)
  CC_TOOL := ccac
  AR_TOOL := arac
  CXX_TOOL := ccac
  LD_TOOL := ccac

  ARC_APP_RUN_CMD = mdb -run -tcf=$(TCF_FILE_NAME) $(DLR)\(DBG_ARGS\)
  ARC_APP_DEBUG_CMD = mdb -OK -tcf=$(TCF_FILE_NAME) $(DLR)\(DBG_ARGS\)

  # The variable TCF_FILE stores path to Tool Configuration File (*.tcf). 
  # This file is used by MWDT toolchain to properly compile/run code
  TCF_FILE ?= 

  LCF_FILE ?= 

  EXT_CFLAGS ?=

  BUILD_ARC_MLI ?= true

# The variable TCF_FILE_NAME stores the TCF file name (including .tcf extension), 
# this variable is used later to add the option to the compiler flags.
# This condition also handles the case when the user/makefile specifies 
# the configuration bundled with MWDT (usually without .tcf extension).
  TCF_FILE_NAME = $(TCF_FILE)

  PLATFORM_FLAGS = -tcf=$(TCF_FILE_NAME) -tcf_core_config
  
  PLATFORM_FLAGS += -Hnocopyr -Hpurge -Hdense_prologue -fslp-vectorize-aggressive -ffunction-sections -fdata-sections 

ifeq ($(filter $(ARC_TAGS), mli20_experimental),)
  PLATFORM_FLAGS += -Hon=Long_enums
endif
  
  # Use compact CRT. It requires pre-defined heap size
  PLATFORM_FLAGS += -Hcl -Hcrt_fast_memcpy -Hcrt_fast_memset 

  PLATFORM_LDFLAGS = -tcf=$(TCF_FILE_NAME) 
  
  PLATFORM_LDFLAGS += -Hnocopyr -m -Hldopt=-Bgrouplib -Hheap=24K 

# Fix for kernel_util for ARC target
  PLATFORM_LDFLAGS += -Hldopt=-u,TfLiteIntArrayEqual -Hldopt=-u,TfLiteIntArrayGetSizeInBytes

ifneq ($(LCF_FILE), )
  PLATFORM_LDFLAGS += $(notdir $(LCF_FILE))
endif

  ldflags_to_remove = -Wl,--fatal-warnings -Wl,--gc-sections
  LDFLAGS := $(filter-out $(ldflags_to_remove),$(LDFLAGS))
  
  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

  CXXFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += $(EXT_CFLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(EXT_CFLAGS)
  LDFLAGS += $(PLATFORM_LDFLAGS)

endif # ARC_TOOLCHAIN

else
  $(error "Only ARC target architecture supported (TARGET_ARCH=arc)")

endif  # TARGET_ARCH
