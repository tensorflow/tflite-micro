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

# ==============================================================================
# 1. Environment and Toolchain Discovery
# ==============================================================================

ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
  $(error "Requires make version 3.82 or later (current is $(MAKE_VERSION))")
endif

# root directory of tensorflow
TENSORFLOW_ROOT :=
RELATIVE_MAKEFILE_DIR := tensorflow/lite/micro/tools/make
MAKEFILE_DIR := $(TENSORFLOW_ROOT)$(RELATIVE_MAKEFILE_DIR)

# Pull in some convenience functions.
include $(MAKEFILE_DIR)/helper_functions.inc

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
  HOST_OS := windows
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Linux)
    HOST_OS := linux
  endif
  ifeq ($(UNAME_S),Darwin)
    HOST_OS := osx
  endif
endif

# Determine the host architecture, with any ix86 architecture being labelled x86_32
HOST_ARCH := $(shell if uname -m | grep -Eq 'i[345678]86'; then echo x86_32; else echo $(shell uname -m); fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# Default compiler and tool names:
TOOLCHAIN := gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

MKDIR_P := mkdir -p
RM_RF := rm -rf

ifeq ($(V),1)
  Q :=
else
  Q := @
endif

ifneq ($(TAGS),)
  $(error The TAGS command line option is no longer supported in the TFLM Makefile.)
endif

# ==============================================================================
# 2. Build Configuration
# ==============================================================================

# Specify which specialized kernel implementation should be pulled in.
OPTIMIZED_KERNEL_DIR :=

# Optimize kernels for speed or memory. This is similar but not the same as KERNEL_OPTIMIZATION_LEVEL and
# CORE_OPTIMIZATION_LEVEL, which specify compiler optimization level.
# Instead this enables a kernel to provide multiple implementations that is configured at build time.
# An example could be a kernel requiring a bigger scratch buffer for certain use cases.
# The example kernel would have a smaller scratch buffer usage when building for size.
# Vice versa it would use more scratch buffer when building for speed and would be more performant.
# Note that this is optional. If having one implementation, nothing needs to be done.
# OPTIMIZE_KERNELS_FOR has only two valid values, KERNELS_OPTIMIZED_FOR_SIZE and KERNELS_OPTIMIZED_FOR_SPEED where the
# former is default.
OPTIMIZE_KERNELS_FOR := KERNELS_OPTIMIZED_FOR_SPEED

# Override this variable from the command line in case the optimized kernels are
# in a different directory.
OPTIMIZED_KERNEL_DIR_PREFIX := $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels
OPTIMIZED_SIGNAL_KERNEL_DIR_PREFIX := $(TENSORFLOW_ROOT)signal/micro/kernels

# Specify which co-processor's kernel implementation should be pulled in.
# If the same kernel is implemented in both kernels/OPTIMIZED_KERNEL_DIR and
# kernels/CO_PROCESSOR, then the implementation from kernels/CO_PROCESSOR will
# be used.
CO_PROCESSOR :=

# This is the way to specify any code path that we want to Include in the build
# process. This will help us to include external code (code that is not part of
# github repo) be tested.
EXTERNAL_DIR :=

# This is the downloads directory inside the makefiles directory
DOWNLOADS_DIR := $(MAKEFILE_DIR)/downloads

INCLUDES := \
-I. \
-I$(DOWNLOADS_DIR) \
-I$(DOWNLOADS_DIR)/gemmlowp \
-I$(DOWNLOADS_DIR)/flatbuffers/include \
-I$(DOWNLOADS_DIR)/kissfft \
-I$(DOWNLOADS_DIR)/ruy

ifneq ($(TENSORFLOW_ROOT),)
  INCLUDES += -I$(TENSORFLOW_ROOT)
endif

ifneq ($(EXTERNAL_DIR),)
  INCLUDES += -I$(EXTERNAL_DIR)
endif

TEST_SCRIPT :=

MICROLITE_LIBS := -lm

# For the optimized_kernel_dir, co-processor and optimize_kernels_for as specified on the
# command line we add -D<tag> to the cflags to allow for #idefs in the code.
#
# We apply the following transformations (via the tr command):
#   1. Convert to uppercase (OPTIMIZED_KERNEL_DIR=xtensa -> -DXTENSA)
ADDITIONAL_DEFINES :=
ifneq ($(OPTIMIZED_KERNEL_DIR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZED_KERNEL_DIR) | tr [a-z] [A-Z])
endif

ifneq ($(CO_PROCESSOR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(CO_PROCESSOR) | tr [a-z] [A-Z])
endif

ifneq ($(OPTIMIZE_KERNELS_FOR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZE_KERNELS_FOR) | tr [a-z] [A-Z])
endif

ifeq ($(TOOLCHAIN), armclang)
  CORE_OPTIMIZATION_LEVEL := -Oz
else
  CORE_OPTIMIZATION_LEVEL := -Os
endif
KERNEL_OPTIMIZATION_LEVEL := -O2
THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL := -O2

# Warn if deprecated optimization level is set.
OPTIMIZATION_LEVEL :=
ifneq ($(OPTIMIZATION_LEVEL),)
$(error "OPTIMIZATION_LEVEL is no longer used.")
endif


CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

ifneq ($(TOOLCHAIN), gcc)
  # GCC can be overly aggressive with shadow warnings, such as warning when a
  # lambda has variable with the same name as a non-captured variable from the
  # enclosing scope. As such, we don't enable shadow warnings on gcc.
  # https://stackoverflow.com/q/66404751
  CC_WARNINGS += -Wshadow
endif

COMMON_FLAGS := \
  -Werror \
  -fno-unwind-tables \
  -fno-asynchronous-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(CC_WARNINGS) \
  $(ADDITIONAL_DEFINES)

ifeq ($(TARGET), $(HOST_OS))
  # If we are not doing a cross-compilation then -DTF_LITE_USE_CTIME is what we
  # want to have by default.
  COMMON_FLAGS += -DTF_LITE_USE_CTIME
endif

CXXFLAGS := \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  $(COMMON_FLAGS)

CCFLAGS := \
  -Wimplicit-function-declaration \
  $(COMMON_FLAGS)

ARFLAGS := -r

ifeq ($(TOOLCHAIN), gcc)
  ifneq ($(TARGET), osx)
    # GCC on MacOS uses an LLVM backend so we avoid the additional linker flags
    # that are unsupported with LLVM.
    LDFLAGS += \
      -Wl,--fatal-warnings \
      -Wl,--gc-sections
  endif
endif

# override these in the makefile.inc for specific compiler targets
TARGET_TOOLCHAIN_PREFIX :=
TARGET_TOOLCHAIN_ROOT :=

# Specifying BUILD_TYPE=<blah> as part of the make command gives us a few
# options to choose from.
#
# If BUILD_TYPE is not specified, the default build (which should be suitable
# most of the time) has all of the error checking logic at the expense of a
# latency increase of ~5-10% relative to BUILD_TYPE=release_with_logs.
#
# This default build is most suited for usual development and testing as is
# highlighted by the discussion on this github pull request:
# https://github.com/tensorflow/tensorflow/pull/42314#issuecomment-694360567
BUILD_TYPE := default
ifeq ($(BUILD_TYPE), debug)
  # Specifying BUILD_TYPE=debug adds debug symbols to the binary (and makes it
  # larger) and should be used to run a binary with gdb.
  CXXFLAGS += -g
  CCFLAGS  += -g
else ifeq ($(BUILD_TYPE), release)
  # The 'release' build results in the smallest binary (by virtue of removing
  # strings from log messages, DCHECKs ...).
  #
  # The down-side is that we currently do not have a good mechanism to allow
  # for logging that is not related to errors (e.g. profiling information, or
  # logs that help determine if tests pass or fail). As a result, we are unable
  # to run tests or benchmarks with BUILD_TYPE=release (which is a bit
  # counter-intuitive). TODO(b/158205789): A global error reporter might help.
  #
  # For a close approximation of the release build use
  # BUILD_TYPE=release_with_logs.
  CXXFLAGS += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
  CCFLAGS  += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
else ifeq ($(BUILD_TYPE), release_with_logs)
  # The latency with BUILD_TYPE=release_with_logs will be close to the 'release'
  # build and there will still be error logs. This build type may be preferable
  # for profiling and benchmarking.
  CXXFLAGS += -DNDEBUG
  CCFLAGS  += -DNDEBUG
else ifeq ($(BUILD_TYPE), no_tf_lite_static_memory)
  # TODO(b/287320282): remove the no_tf_lite_static_memory build.
  #
  # This build should not be used to run any binaries/tests since
  # TF_LITE_STATIC_MEMORY should be defined for all micro builds. However,
  # having a build without TF_LITE_STATIC_MEMORY is useful to catch errors in
  # code that is shared between TfLite Mobile and TfLite Micro. See this issue
  # for more details:
  # https://github.com/tensorflow/tensorflow/issues/43076
  CXXFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CXXFLAGS))
  CCFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CCFLAGS))
endif

ifeq ($(CC_VER11), true)
  CXXFLAGS += -std=c++11
  CCFLAGS += -std=c11
else
  CXXFLAGS += -std=c++17
  CCFLAGS += -std=c17
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
MICROLITE_LIB_NAME := libtensorflow-microlite.a

# TFLM optional compression support (default disabled)
ENABLE_COMPRESSION := no
ifneq ($(USE_TFLM_COMPRESSION),)
  # currently only Linux (x86), Xtensa targets supported
  ifeq ($(TARGET), $(filter $(TARGET), linux xtensa))
    CXXFLAGS += -DUSE_TFLM_COMPRESSION
    CCFLAGS += -DUSE_TFLM_COMPRESSION
    ENABLE_COMPRESSION := yes
  endif
endif

# ==============================================================================
# 3. Directory Definitions
# ==============================================================================

# Where compiled objects are stored.
BASE_GENDIR := gen
GENDIR := $(BASE_GENDIR)/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)
ifneq ($(OPTIMIZED_KERNEL_DIR),)
  GENDIR := $(GENDIR)_$(OPTIMIZED_KERNEL_DIR)
endif
ifneq ($(CO_PROCESSOR),)
  GENDIR := $(GENDIR)_$(CO_PROCESSOR)
endif
ifeq ($(ENABLE_COMPRESSION), yes)
  GENDIR := $(GENDIR)_compression
endif
GENDIR := $(GENDIR)_$(TOOLCHAIN)/

CORE_OBJDIR := $(GENDIR)obj/core/
KERNEL_OBJDIR := $(GENDIR)obj/kernels/
THIRD_PARTY_KERNEL_OBJDIR := $(GENDIR)obj/third_party_kernels/
THIRD_PARTY_OBJDIR := $(GENDIR)obj/third_party/
GENERATED_SRCS_DIR := $(GENDIR)genfiles/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/
PRJDIR := $(GENDIR)prj/

# ==============================================================================
# 4. Source File Definitions
# ==============================================================================

# TODO(b/152645559): move all benchmarks to benchmarks directory.
include $(MAKEFILE_DIR)/sources.inc

# ==============================================================================
# 5. Third-Party & Specialization Logic
# ==============================================================================

include $(MAKEFILE_DIR)/third_party.inc

MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_TEST_SRCS), $(MICROLITE_CC_BASE_SRCS))
MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_BENCHMARK_SRCS), $(MICROLITE_CC_SRCS))

# The target-specific makefile must have a name that is exactly
# TARGET_makefile.inc and is only needed for cross-compilation (i.e. when TARGET
# is different from the HOST_OS).
TARGETS_WITHOUT_MAKEFILES := \
$(HOST_OS)

# This specific string needs to be outputted for a test to be recognized as
# having passed.
TEST_PASS_STRING := '~~~ALL TESTS PASSED~~~'

# $(TARGET)_makefile.inc can set this to true to allow it to defined a custom
# implementation for `make test`. See bluepill_makefile as an example.
TARGET_SPECIFIC_MAKE_TEST := 0

ifeq ($(findstring $(TARGET),$(TARGETS_WITHOUT_MAKEFILES)),)
  include $(MAKEFILE_DIR)/targets/$(TARGET)_makefile.inc
endif

# Create rules for downloading third-party dependencies.
THIRD_PARTY_TARGETS :=
$(foreach DOWNLOAD,$(THIRD_PARTY_DOWNLOADS),$(eval $(call create_download_rule,$(DOWNLOAD))))
third_party_downloads: $(THIRD_PARTY_TARGETS)

# Validate valid options.
ifeq (,$(filter $(OPTIMIZE_KERNELS_FOR),KERNELS_OPTIMIZED_FOR_SPEED KERNELS_OPTIMIZED_FOR_SIZE))
    $(error Incorrect OPTIMIZE_KERNELS_FOR: $(OPTIMIZE_KERNELS_FOR))
endif

ifneq ($(OPTIMIZED_KERNEL_DIR),)
  PATH_TO_OPTIMIZED_KERNELS := $(OPTIMIZED_KERNEL_DIR_PREFIX)/$(OPTIMIZED_KERNEL_DIR)
  PATH_TO_SIGNAL_OPTIMIZED_KERNELS := $(OPTIMIZED_SIGNAL_KERNEL_DIR_PREFIX)/$(OPTIMIZED_KERNEL_DIR)

  # Check that OPTIMIZED_KERNEL_DIR is valid to avoid unexpected fallback to
  # reference kernels. See http://b/183546742 for more context.
  RESULT := $(shell $(MAKEFILE_DIR)/check_optimized_kernel_dir.sh $(PATH_TO_OPTIMIZED_KERNELS))
  ifneq ($(RESULT), SUCCESS)
    $(error Incorrect OPTIMIZED_KERNEL_DIR: $(RESULT))
  endif

  include $(MAKEFILE_DIR)/ext_libs/$(OPTIMIZED_KERNEL_DIR).inc
  # Specialize for the optimized kernels
  MICROLITE_CC_KERNEL_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
		--base_files "$(MICROLITE_CC_KERNEL_SRCS)" \
		--specialize_directory $(PATH_TO_OPTIMIZED_KERNELS))

  ifneq ($(filter $(OPTIMIZED_KERNEL_DIR), xtensa hexagon),)
    # Check that OPTIMIZED_KERNEL_DIR is valid to avoid unexpected fallback to
    # reference kernels. See http://b/183546742 for more context.
    RESULT := $(shell $(MAKEFILE_DIR)/check_optimized_kernel_dir.sh $(PATH_TO_SIGNAL_OPTIMIZED_KERNELS))
    ifneq ($(RESULT), SUCCESS)
      $(error Incorrect SIGNAL OPTIMIZED_KERNEL_DIR: $(RESULT))
    endif

    # Specialize for the optimized kernels
    MICROLITE_CC_SIGNAL_KERNEL_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
      --base_files "$(MICROLITE_CC_SIGNAL_KERNEL_SRCS)" \
      --specialize_directory $(PATH_TO_SIGNAL_OPTIMIZED_KERNELS))
    MICROLITE_CC_KERNEL_SRCS += $(wildcard $(PATH_TO_SIGNAL_OPTIMIZED_KERNELS)/*.S)
    MICROLITE_CC_HDRS += $(wildcard $(PATH_TO_SIGNAL_OPTIMIZED_KERNELS)/*.h)
  endif

  # The first ifneq is needed to be compatible with make versions prior to 4.2
  # which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
  # Ubuntu 18.04 only has version 4.1
  ifneq ($(.SHELLSTATUS),)
    ifneq ($(.SHELLSTATUS),0)
      $(error Error with specialize_files.py $(MICROLITE_CC_KERNEL_SRCS))
    endif
  endif

  # Optimized kernel directories can have their own header files which need to
  # be included in MICROLITE_CC_HDRS for project generation to have a complete
  # list of headers.
  MICROLITE_CC_HDRS += $(wildcard $(PATH_TO_OPTIMIZED_KERNELS)/*.h)
endif

MICROLITE_CC_KERNEL_SRCS += $(MICROLITE_CC_SIGNAL_KERNEL_SRCS)

# If a co-processor is specified on the command line with
# CO_PROCESSOR=<co_processor> then we will include ext_libs/<co_processor>.inc
# and find additional kernel sources in kernels/<co_processor>/
#
# That the co-processor specialization of the kernel sources happens after the
# optimized_kernel_dir means that if there is an implementation of the same
# kernel in both directories, the one from co_processor will be used.
ifneq ($(CO_PROCESSOR),)
  include $(MAKEFILE_DIR)/ext_libs/$(CO_PROCESSOR).inc
  # Specialize for the coprocessor kernels.
  PATH_TO_COPROCESSOR_KERNELS := $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/$(CO_PROCESSOR)
  MICROLITE_CC_KERNEL_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
    --base_files "$(MICROLITE_CC_KERNEL_SRCS)" \
    --specialize_directory $(PATH_TO_COPROCESSOR_KERNELS))

  # The first ifneq is needed to be compatible with make versions prior to 4.2
  # which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
  # Ubuntu 18.04 only has version 4.1
  ifneq ($(.SHELLSTATUS),)
    ifneq ($(.SHELLSTATUS),0)
      $(error Error with specialize_files.py $(MICROLITE_CC_KERNEL_SRCS))
    endif
  endif
endif

# Specialize for debug_log. micro_time etc.
PATH_TO_TARGET_SRCS := $(TENSORFLOW_ROOT)tensorflow/lite/micro/$(TARGET)
MICROLITE_CC_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
  --base_files "$(MICROLITE_CC_SRCS)" \
  --specialize_directory $(PATH_TO_TARGET_SRCS))

# The first ifneq is needed to be compatible with make versions prior to 4.2
# which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
# Ubuntu 18.04 only has version 4.1
ifneq ($(.SHELLSTATUS),)
  ifneq ($(.SHELLSTATUS),0)
    $(error Error with specialize_files.py $(MICROLITE_CC_SRCS))
  endif
endif

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_CC_KERNEL_SRCS) \
	$(MICROLITE_TEST_SRCS)

# ==============================================================================
# 6. Build Rules and Targets
# ==============================================================================

.PHONY: all microlite build test integration_tests generated_micro_mutable_op_resolver \
        clean clean_downloads list_gendir list_library_sources list_library_headers \
        list_third_party_sources list_third_party_headers list_generator_dir \
        third_party_downloads

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)$(CXX_TOOL)
CC := $(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)$(CC_TOOL)
AR := $(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)$(AR_TOOL)

# The default Makefile target(all) must appear before any target,
# which is compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH)

# Include output directory since example cc files depend on generated headers.
INCLUDES += -I$(GENERATED_SRCS_DIR)
INCLUDES += -I$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)

# Load test and benchmark definitions.
include $(MAKEFILE_DIR)/tests.inc

MICROLITE_LIB_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_SRCS)))))

MICROLITE_THIRD_PARTY_OBJS := $(addprefix $(THIRD_PARTY_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(THIRD_PARTY_CC_SRCS)))))

MICROLITE_THIRD_PARTY_KERNEL_OBJS := $(addprefix $(THIRD_PARTY_KERNEL_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(THIRD_PARTY_KERNEL_CC_SRCS)))))

MICROLITE_KERNEL_OBJS := $(addprefix $(KERNEL_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_KERNEL_SRCS)))))

$(CORE_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CXX) $(CXXFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(CORE_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(CORE_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CXX) $(CXXFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_KERNEL_OBJDIR)%.o: %.cc $(THIRD_PARTY_KERNEL_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CXX) $(CXXFLAGS) $(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_KERNEL_OBJDIR)%.o: %.c $(THIRD_PARTY_KERNEL_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_KERNEL_OBJDIR)%.o: %.S $(THIRD_PARTY_KERNEL_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CXX) $(CXXFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CC) $(CCFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

microlite: $(MICROLITE_LIB_PATH)

# Gathers together all the objects we've compiled into a single '.a' archive.
$(MICROLITE_LIB_PATH): $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS) $(MICROLITE_THIRD_PARTY_OBJS) $(MICROLITE_THIRD_PARTY_KERNEL_OBJS) $(MICROLITE_CUSTOM_OP_OBJS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS) \
	$(MICROLITE_KERNEL_OBJS) $(MICROLITE_THIRD_PARTY_OBJS) $(MICROLITE_THIRD_PARTY_KERNEL_OBJS) $(MICROLITE_CUSTOM_OP_OBJS)

$(BINDIR)%_test : $(CORE_OBJDIR)%_test.o $(MICROLITE_LIB_PATH)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $< \
	$(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

$(BINDIR)%.test_target: $(BINDIR)%_test
	@test -f $(TEST_SCRIPT) || (echo 'Unable to find the test script. Is the software emulation available in $(TARGET)?'; exit 1)
	$(TEST_SCRIPT) $< $(TEST_PASS_STRING)

# snease: Add %.bin rule here since BINDIR is now defined
# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
ifeq ($(TOOLCHAIN), armclang)
  CXXFLAGS += -ffp-mode=full
  FROMELF := $(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)fromelf
  $(BINDIR)%.bin: $(BINDIR)%
		$(Q)$(MKDIR_P) $(dir $@)
		$(Q)$(FROMELF) --bin --output=$@ $<
else
  OBJCOPY := $(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)%.bin: $(BINDIR)%
		$(Q)$(MKDIR_P) $(dir $@)
		$(Q)$(OBJCOPY) $< $@ -O binary
endif

# Create kernel test targets.
include $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/Makefile.inc

# Create binary size test target.
include $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/ci_build/binary_size_test/Makefile.inc



ifeq ($(TARGET_SPECIFIC_MAKE_TEST),0)
test: $(MICROLITE_TEST_TARGETS)
integration_tests: $(MICROLITE_INTEGRATION_TEST_TARGETS)
generated_micro_mutable_op_resolver: $(MICROLITE_GEN_OP_RESOLVER_TEST_TARGETS)
endif

# Just build the test targets
build: $(MICROLITE_BUILD_TARGETS)

list_gendir:
	@echo $(GENDIR)

list_library_sources:
	@echo $(MICROLITE_CC_SRCS) $(MICROLITE_CC_KERNEL_SRCS)

list_library_headers:
	@echo $(MICROLITE_CC_HDRS)

list_third_party_sources:
	@echo $(THIRD_PARTY_CC_SRCS) $(THIRD_PARTY_KERNEL_CC_SRCS)

list_third_party_headers:
	@echo $(THIRD_PARTY_CC_HDRS)

list_generator_dir:
	@echo $(GENERATED_SRCS_DIR)

# Gets rid of all generated files.
clean:
	$(RM_RF) $(BASE_GENDIR)

# Removes third-party downloads.
clean_downloads:
	$(RM_RF) $(DOWNLOADS_DIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))

# Target to print build configuration info for debugging.
config_info:
	@echo "Build Configuration:"
	@echo "  TARGET: $(TARGET)"
	@echo "  TARGET_ARCH: $(TARGET_ARCH)"
	@echo "  TOOLCHAIN: $(TOOLCHAIN)"
	@echo "  BUILD_TYPE: $(BUILD_TYPE)"
	@echo "  TENSORFLOW_ROOT: $(TENSORFLOW_ROOT)"
	@echo "  EXTERNAL_DIR: $(EXTERNAL_DIR)"
	@echo "  DOWNLOADS_DIR: $(DOWNLOADS_DIR)"
	@echo "  GENDIR: $(GENDIR)"
	@echo "  OPTIMIZED_KERNEL_DIR: $(OPTIMIZED_KERNEL_DIR)"
	@echo "  CO_PROCESSOR: $(CO_PROCESSOR)"
	@echo "  TARGET_TOOLCHAIN_ROOT: $(TARGET_TOOLCHAIN_ROOT)"
	@echo "  INCLUDES: $(INCLUDES)"
	@echo "  THIRD_PARTY_DOWNLOADS: $(THIRD_PARTY_DOWNLOADS)"
	@echo "  MICROLITE_BENCHMARK_SRCS: $(MICROLITE_BENCHMARK_SRCS)"
	@echo "  MICROLITE_TEST_SRCS: $(MICROLITE_TEST_SRCS)"
