# Settings for RISCV 32-bit MCU toolchain.
TARGET_ARCH := riscv32_mcu
TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

export PATH := $(MAKEFILE_DIR)/downloads/riscv_toolchain/bin/:$(PATH)

$(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))
$(eval $(call add_third_party_download,$(SIFIVE_FE310_LIB_URL),$(SIFIVE_FE310_LIB_MD5),sifive_fe310_lib,patch_sifive_sdk))

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/renode_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the renode download: $(DOWNLOAD_RESULT))
endif

PLATFORM_FLAGS = \
  -march=rv32imac \
  -mabi=ilp32 \
  -mcmodel=medany \
  -mexplicit-relocs \
  -fno-builtin-printf \
  -fno-exceptions \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
  -funsigned-char \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer

CXXFLAGS += $(PLATFORM_FLAGS) \
  -fpermissive \
  -fno-rtti \
  -fno-threadsafe-statics \
  -fno-use-cxa-atexit \
  -DTF_LITE_USE_GLOBAL_MIN \
  -DTF_LITE_USE_GLOBAL_MAX \
  --std=gnu++11

CCFLAGS += $(PLATFORM_FLAGS)

BUILD_TYPE := micro

INCLUDES += \
  -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/include \
  -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/drivers/ \
  -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
  -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1

MICRO_FE310_BSP_ENV_SRCS := \
  $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/start.S \
  $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/entry.S \
  $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/init.c

MICRO_FE310_LIBWRAP_SRCS := \
  $(wildcard $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/sys/*.c) \
  $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/misc/write_hex.c \
  $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/stdlib/malloc.c

MICROLITE_CC_SRCS += \
  $(wildcard tensorflow/lite/micro/riscv32_mcu/*.cc) \
  $(MICRO_FE310_BSP_ENV_SRCS) \
  $(MICRO_FE310_LIBWRAP_SRCS)

LIBWRAP_SYMS := malloc free \
                open lseek read write fstat stat close link unlink \
                execve fork getpid kill wait \
                isatty times sbrk _exit puts
LDFLAGS += \
  -T$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/flash-256k.lds \
  -L$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
  --specs=nano.specs

# Use startup files from BSP
LDFLAGS += -nostartfiles
# Don't link against standard libraries, especially libstdc++
LDFLAGS += -nostdlib

# Wrap some standard symbols for which the BSP provides wrappers
LDFLAGS += \
  $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=$(s)) \
  $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=_$(s))

# Explicitly link against libraries excluded by -nostdlib, note the lack of libstdc++
LDFLAGS += -Wl,--start-group -lm -lc_nano -lgcc $(MICROLITE_LIB_PATH) -Wl,--end-group

# This disables the "linker relaxation" optimization, which produced incorrect code.
# TODO: Check whether this is fixed in newer versions of the toolchain.
LDFLAGS += -mno-relax

# flatbuffer_utils_test and micro_allocator_test need to link against libstdc++.
# This is problematic because our toolchain's libstdc++ was compiled with exceptions,
# so when we try to link against it, the linker tries to put the exception handling
# related sections in the binary, which we don't want.
#
# The other two excluded tests also fail on other platforms, so disable them for now.
EXCLUDED_TESTS := \
  tensorflow/lite/micro/flatbuffer_utils_test.cc \
  tensorflow/lite/micro/micro_interpreter_test.cc \
  tensorflow/lite/micro/memory_arena_threshold_test.cc \
  tensorflow/lite/micro/micro_allocator_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

TEST_SCRIPT := tensorflow/lite/micro/testing/test_with_renode.sh

# We are setting this variable to non-zero to allow us to have a custom
# implementation of `make test` for our target
TARGET_SPECIFIC_MAKE_TEST := 1

# Exclude this test from `make test`, as it takes a very long time (~8 minutes on a regular PC)
TEST_TARGET_EXCLUDED_BINARIES = $(BINDIR)/simple_features_generator_test

ALL_TEST_TARGET_BINARIES = $(shell ls -1 $(BINDIR)/*_test)

TEST_TARGET_BINARIES = $(filter-out $(TEST_TARGET_EXCLUDED_BINARIES),$(ALL_TEST_TARGET_BINARIES))

test: build
	$(TEST_SCRIPT) "$(TEST_TARGET_BINARIES)" $(TEST_PASS_STRING) $(TARGET)

# This target runs all tests unconditionally
test_all: build
	$(TEST_SCRIPT) "$(ALL_TEST_TARGET_BINARIES)" $(TEST_PASS_STRING) $(TARGET)
