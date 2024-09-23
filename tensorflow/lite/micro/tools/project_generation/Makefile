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

ifneq ($(TENSORFLOW_ROOT),)
  INCLUDES += -I$(TENSORFLOW_ROOT)
endif

ifneq ($(EXTERNAL_DIR),)
  INCLUDES += -I$(EXTERNAL_DIR)
endif

ifeq ($(BUILD_TYPE), cmsis_nn)
  CXX := arm-none-eabi-g++
  CC := arm-none-eabi-gcc
  AR := arm-none-eabi-ar

  INCLUDES += \
    -I./third_party/cmsis \
    -I./third_party/cmsis_nn \
    -I./third_party/cmsis_nn/Include \
    -I./third_party/cmsis/CMSIS/Core/Include

  COMMON_FLAGS += \
    -DTF_LITE_MCU_DEBUG_LOG \
    -DPROJECT_GENERATION \
    -mthumb \
    -mlittle-endian \
    -funsigned-char \
    -fomit-frame-pointer \
    -MD \
    -DCMSIS_NN

endif

CXXFLAGS := \
  -std=c++17 \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  $(COMMON_FLAGS)

CCFLAGS := \
  -std=c17 \
  $(COMMON_FLAGS)

ARFLAGS := -r

GENDIR := gen
OBJDIR := $(GENDIR)/obj
BINDIR := $(GENDIR)/bin
LIB := $(GENDIR)/libtflm.a

TFLM_CC_SRCS := $(shell find $(TENSORFLOW_ROOT)tensorflow -name "*.cc" -o -name "*.c")
TFLM_CC_SRCS += $(shell find $(TENSORFLOW_ROOT)signal -name "*.cc" -o -name "*.c")
OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.c,%.o,$(patsubst %.cc,%.o,$(TFLM_CC_SRCS))))

# if the third party printf library is present, add the include paths
TFLM_PRINTF_PATH := $(shell find third_party -name eyalroz_printf)
ifneq ($(TFLM_PRINTF_PATH),)
  INCLUDES += \
    -I./third_party
endif

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

HELLO_WORLD_SRCS := $(wildcard examples/hello_world/*.cc)
HELLO_WORLD_SRCS += $(wildcard examples/hello_world/models/*.cc)
HELLO_WORLD_INCLUDES := $(INCLUDES) -I./examples/hello_world

hello_world: libtflm
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(HELLO_WORLD_SRCS) $(HELLO_WORLD_INCLUDES) $(LIB) -o $(BINDIR)/$@

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

examples: hello_world micro_speech person_detection
