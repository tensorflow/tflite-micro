# Settings for RISCV 32-bit toolchain.
TARGET_ARCH := riscv32
TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

RISCV_ARCH := rv32imc_zve32x_zvl128b
RISCV_ABI := ilp32
RISCV_CODE_MODEL := medany

# Allow additional flags on the command line for debugging.
RISCV_EXTRA_CFLAGS :=

TARGET_DEFAULT_TOOLCHAIN_ROOT := $(HOME)/rv32imc_zve32x_zvl128b/bin/
TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
  $(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))
endif

export PATH := $(TARGET_TOOLCHAIN_ROOT):$(PATH)

PLATFORM_FLAGS = \
  -march=$(RISCV_ARCH) \
  -mabi=$(RISCV_ABI) \
  -mcmodel=$(RISCV_CODE_MODEL) \
  -mexplicit-relocs \
  -fno-builtin-printf \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
  -funsigned-char \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer \
  -DTFLM_USE_RISCV_VECTOR

CXXFLAGS += $(PLATFORM_FLAGS) \
  -fpermissive \
  -fno-use-cxa-atexit \
  -DTF_LITE_USE_GLOBAL_MIN \
  -DTF_LITE_USE_GLOBAL_MAX

CCFLAGS += $(PLATFORM_FLAGS)

BUILD_TYPE := micro

LDFLAGS += --specs=nano.specs

# See http://b/15851472 for why memory arena threshold test is disabled.
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

CCFLAGS += $(RISCV_EXTRA_CFLAGS)
CXXFLAGS += $(RISCV_EXTRA_CFLAGS)

# This disables the "linker relaxation" optimization, which produced incorrect code.
# TODO(b/279805615): Check whether this is fixed in newer versions of the toolchain.
LDFLAGS += -mno-relax
TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_spike.sh
SIZE_SCRIPT := ${TENSORFLOW_ROOT}tensorflow/lite/micro/testing/size_riscv32_binary.sh

include $(MAKEFILE_DIR)/ext_libs/eyalroz_printf.inc

MICROLITE_CC_SRCS += \
  tensorflow/lite/micro/kernels/riscv_vector/conv_rvv.cc \
  tensorflow/lite/micro/kernels/riscv_vector/fully_connected_rvv.cc \
  tensorflow/lite/micro/kernels/riscv_vector/conv.cc \
  tensorflow/lite/micro/kernels/riscv_vector/depthwise_conv.cc \
  tensorflow/lite/micro/kernels/riscv_vector/fully_connected.cc \
  tensorflow/lite/micro/kernels/riscv_vector/pooling.cc \
  tensorflow/lite/micro/kernels/riscv_vector/pooling_rvv.cc \
  tensorflow/lite/micro/kernels/riscv_vector/signal/rfft.cc\
  tensorflow/lite/micro/kernels/riscv_vector/signal/rfft_int16_rvv.cc \
  tensorflow/lite/micro/kernels/riscv_vector/signal/filter_bank.cc \
  tensorflow/lite/micro/kernels/riscv_vector/signal/filter_bank_rvv.cc \
#  tensorflow/lite/micro/kernels/riscv_vector/softmax.cc \

EXCLUDED_SRCS := \
  tensorflow/lite/micro/kernels/conv.cc \
  tensorflow/lite/micro/kernels/depthwise_conv.cc \
  tensorflow/lite/micro/kernels/fully_connected.cc \
  tensorflow/lite/micro/kernels/pooling.cc\
  signal/micro/kernels/rfft.cc \
  signal/src/rfft_int16.cc \
  signal/src/kiss_fft_wrappers/kiss_fft_int16.cc \
  signal/micro/kernels/filter_bank.cc \
  signal/src/filter_bank.cc \
#  tensorflow/lite/micro/kernels/softmax.cc \


