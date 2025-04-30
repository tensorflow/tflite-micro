# Settings for Xtensa toolchain for the kernels.
# REQUIRED:
#  Environment variables:
#   - XTENSA_BASE  must be set to location of
#     the Xtensa developer tools installation directory.
#  Command line arguments:
#   - XTENSA_TOOLS_VERSION: For example: RI-2019.2-linux
#   - XTENSA_CORE: The name of the Xtensa core to use
#      For example: HIFI_190304_swupgrade

TARGET_ARCH :=
XTENSA_USE_LIBC :=

# Allow additional flags on the command line for debugging.
XTENSA_EXTRA_CFLAGS :=

ifndef XTENSA_BASE
  $(error XTENSA_BASE is undefined)
endif

ifndef XTENSA_TOOLS_VERSION
  $(error XTENSA_TOOLS_VERSION is undefined)
endif

ifndef XTENSA_CORE
  $(error XTENSA_CORE is undefined)
endif

ifeq ($(TARGET_ARCH), )
  $(error TARGET_ARCH must be specified on the command line)
endif

# Create a cflag based on the specified TARGET_ARCH. For example:
#   TARGET_ARCH=hifi4 --> -DHIFI4
TARGET_ARCH_DEFINES := -D$(shell echo $(TARGET_ARCH) | tr [a-z] [A-Z])

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_CTIME \
  --xtensa-core=$(XTENSA_CORE) \
  -mcoproc \
  $(TARGET_ARCH_DEFINES) \
  -mlongcalls

export PATH := $(XTENSA_BASE)/tools/$(XTENSA_TOOLS_VERSION)/XtensaTools/bin:$(PATH)
TARGET_TOOLCHAIN_PREFIX := xt-
CXX_TOOL := clang++
CC_TOOL := clang

# Building with C++17 requires libc++
ifneq ($(CC_VER11), true)
  PLATFORM_FLAGS += -stdlib=libc++
endif

# Unused exception related symbols make their way into a binary that links
# against TFLM as described in https://github.com/tensorflow/tensorflow/issues/47575.
# We have two options to avoid this. The first involves using -stdlib=libc++ and
# the second involves stubbing out and modifying some of the files in the Xtensa
# toolchain to prevent inclusion of the exception handling code
# (http://b/182209217#comment3). This Makefile supports building TFLM in a way
# that is compatible with either of the two approaches.
ifeq ($(XTENSA_USE_LIBC), true)
  PLATFORM_FLAGS += -stdlib=libc++
else
  # TODO(b/150240249): Do not filter-out -fno-rtti once that works for the
  # Xtensa toolchain.
  CXXFLAGS := $(filter-out -fno-rtti, $(CXXFLAGS))
endif

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

CCFLAGS += $(XTENSA_EXTRA_CFLAGS)
CXXFLAGS += $(XTENSA_EXTRA_CFLAGS)

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_xtensa_binary.sh
SIZE_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/size_xtensa_binary.sh

# TODO(b/158651472): Fix the memory_arena_threshold_test
# TODO(b/174707181): Fix the micro_interpreter_test
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

# TODO(b/156962140): This manually maintained list of excluded examples is
# quite error prone.
EXCLUDED_EXAMPLE_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/Makefile.inc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_recognition_experimental/Makefile.inc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/network_tester/Makefile.inc
MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))
MICRO_LITE_EXAMPLE_TESTS += $(shell find $(TENSORFLOW_ROOT)third_party/xtensa/examples/ -name Makefile.inc)

# Needed for LSTM support.
MICROLITE_CC_KERNEL_SRCS := $(MICROLITE_CC_KERNEL_SRCS) \
$(TENSORFLOW_ROOT)tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/kernels/kernel_util.cc

ifeq ($(OPTIMIZED_KERNEL_DIR), xtensa)
  MICROLITE_CC_KERNEL_SRCS := $(MICROLITE_CC_KERNEL_SRCS) \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/lstm_eval.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/lstm_eval_hifi.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/unidirectional_sequence_lstm.cc

  # override KERNEL_OPTIMIZATION_LEVEL to enable higher performance
  # Xtensa intrinsics.
$(KERNEL_OBJDIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/decompress.o: $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/decompress.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -O3 -LNO:simd $(INCLUDES) -c $< -o $@
endif
