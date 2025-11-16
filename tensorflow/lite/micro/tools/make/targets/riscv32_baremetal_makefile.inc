# Toolchain Configuration
TARGET_ARCH := riscv32
TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

# Toolchain Path 
TARGET_TOOLCHAIN_ROOT := /home/mips/riscv32/bin/

# Minimum necessary flags to avoid pulling unnecessary C++ runtime 
# Support and for using lightweight max and min 
CXXFLAGS += -fno-use-cxa-atexit  \
            -DTF_LITE_USE_GLOBAL_MIN  \
            -DTF_LITE_USE_GLOBAL_MAX \

# Minimum necessary flags for ensuring our custom startup file is included and not 
# the default one and for linking our linker to incorporate it in the build 
LDFLAGS = -nostartfiles  \
          -T tensorflow/lite/micro/riscv32_baremetal/linker.ld 

# Include c standard library, math library and compiler support library 
MICROLITE_LIBS = -lc -lm -lgcc 

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_qemu.sh

# List of available examples
EXAMPLES := person_detection hello_world micro_speech person_detection_test hello_world_test micro_speech_test dtln_test

# Run a given example in QEMU using test_with_qemu.sh
$(EXAMPLES):
	@$(TEST_SCRIPT) system-riscv32 - gen/riscv32_baremetal_x86_64_default_gcc/bin/$@ non_test_binary riscv32_baremetal


