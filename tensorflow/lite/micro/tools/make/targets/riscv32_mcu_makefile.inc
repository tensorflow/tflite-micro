# Settings for RISCV 32-bit MCU toolchain.
TARGET_ARCH := riscv32_mcu
TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

export PATH := $(MAKEFILE_DIR)/downloads/riscv_toolchain/bin/:$(PATH)

$(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))
$(eval $(call add_third_party_download,$(SIFIVE_FE310_LIB_URL),$(SIFIVE_FE310_LIB_MD5),sifive_fe310_lib,))

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
  -T$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/flash.lds \
  -L$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
  --specs=nano.specs

# Wrap some standard symbols for which the BSP provides wrappers
LDFLAGS += \
  $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=$(s)) \
  $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=_$(s))

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy
$(BINDIR)/%.bin: $(BINDIR)/%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary
