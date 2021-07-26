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
#
# Simple Makefile that serves as a smokes-check for project generation on x86.
#
# Execute the following command after copying this Makefile to the root of the
# TFLM tree created with the project generation script:
# make -j8 examples

CXX := clang++
CXXFLAGS := \
  -std=c++11

CC := clang
CCFLAGS := \
  -std=c11

INCLUDES := \
  -I. \
  -I./third_party/gemmlowp \
  -I./third_party/flatbuffers/include \
  -I./third_party/kissfft \
  -I./third_party/kissfft/tools \
  -I./third_party/ruy

AR := ar
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
MICRO_SPEECH_THIRD_PARTY_SRCS := $(wildcard third_party/kissfft/*.c)
MICRO_SPEECH_THIRD_PARTY_SRCS += $(wildcard third_party/kissfft/*/*.c)

micro_speech: libtflm
	@mkdir -p $(BINDIR)
# some includes are relative to the repo location of the header file
# so a symbolic link is created to compensate
	@ln -sfn ../../../examples tensorflow/lite/micro/examples
	$(CXX) $(CXXFLAGS) $(MICRO_SPEECH_SRCS) $(MICRO_SPEECH_THIRD_PARTY_SRCS) $(INCLUDES) $(LIB) -o $(BINDIR)/$@

PERSON_DETECTION_SRCS := $(wildcard examples/person_detection/*.cc)
PERSON_DETECTION_THIRD_PARTY_SRCS := $(wildcard third_party/person_model_int8/*.cc)

person_detection: libtflm
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(PERSON_DETECTION_SRCS) $(PERSON_DETECTION_THIRD_PARTY_SRCS) $(INCLUDES) $(LIB) -o $(BINDIR)/$@

examples: hello_world magic_wand micro_speech person_detection
