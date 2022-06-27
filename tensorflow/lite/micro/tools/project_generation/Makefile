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
#
# Simple Makefile that serves as a smokes-check for project generation on x86.
#
# Execute the following command after copying this Makefile to the root of the
# TFLM tree created with the project generation script:
# make -j8 examples

BUILD_TYPE :=

COMMON_FLAGS := \
  -DTF_LITE_STATIC_MEMORY \
  -fno-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0

CXX := clang++
CC := clang
AR := ar

INCLUDES := \
  -I. \
  -I./third_party/gemmlowp \
  -I./third_party/flatbuffers/include \
  -I./third_party/kissfft \
  -I./third_party/kissfft/tools \
  -I./third_party/ruy

CXXFLAGS :=
CCFLAGS :=

ifeq ($(BUILD_TYPE), cmsis_nn)
  CXX := arm-none-eabi-g++
  CC := arm-none-eabi-gcc
  AR := arm-none-eabi-ar

  INCLUDES += \
    -I./third_party/cmsis \
    -I./third_party/cmsis/CMSIS/Core/Include \
    -I./third_party/cmsis/CMSIS/DSP/Include \
    -I./third_party/cmsis/CMSIS/NN/Include

  COMMON_FLAGS += \
    -DTF_LITE_MCU_DEBUG_LOG \
    -mthumb \
    -mlittle-endian \
    -funsigned-char \
    -fomit-frame-pointer \
    -MD \
    -DCMSIS_NN

  # For this build type there is additional dependency to CMSIS.
  TOOLCHAIN = gcc
  SKIP_ARM_GCC_DOWNLOAD := "yes please"
  include cortex_m_generic_makefile.inc
  INCLUDES += \
    -I./third_party/cmsis/Device/ARM/$(ARM_CPU)/Include

endif

CXXFLAGS += \
  -std=c++11 \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  $(COMMON_FLAGS)

CCFLAGS += \
  -std=c11 \
  $(COMMON_FLAGS)

ARFLAGS := -r

GENDIR := gen
OBJDIR := $(GENDIR)/obj
BINDIR := $(GENDIR)/bin
LIB := $(GENDIR)/libtflm.a

TFLM_CC_SRCS := $(shell find tensorflow -name "*.cc" -o -name "*.c")
OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.c,%.o,$(patsubst %.cc,%.o,$(TFLM_CC_SRCS))))

$(OBJDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB) $(OBJS)

clean:
	rm -rf $(GENDIR)

libtflm: $(LIB)

hello_world: libtflm
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(wildcard examples/hello_world/*.cc) $(INCLUDES) $(LIB) -o $(BINDIR)/$@

magic_wand: libtflm
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(wildcard examples/magic_wand/*.cc) $(INCLUDES) $(LIB) -o $(BINDIR)/$@

MICRO_SPEECH_SRCS := $(wildcard examples/micro_speech/*.cc)
MICRO_SPEECH_SRCS += $(wildcard examples/micro_speech/*/*.cc)
MICRO_SPEECH_THIRD_PARTY_SRCS :=
MICRO_SPEECH_INCLUDES := $(INCLUDES) -I./examples/micro_speech

micro_speech: libtflm
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(MICRO_SPEECH_SRCS) $(MICRO_SPEECH_THIRD_PARTY_SRCS) $(MICRO_SPEECH_INCLUDES) $(LIB) -o $(BINDIR)/$@

PERSON_DETECTION_SRCS := $(wildcard examples/person_detection/*.cc)
PERSON_DETECTION_THIRD_PARTY_SRCS := $(wildcard third_party/person_model_int8/*.cc)
PERSON_DETECTION_INCLUDES := $(INCLUDES) -I./examples/person_detection

person_detection: libtflm
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(PERSON_DETECTION_SRCS) $(PERSON_DETECTION_THIRD_PARTY_SRCS) $(PERSON_DETECTION_INCLUDES) $(LIB) -o $(BINDIR)/$@

examples: hello_world magic_wand micro_speech person_detection
