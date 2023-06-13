TARGET_DEFAULT_TOOLCHAIN_ROOT := 
TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
TARGET_TOOLCHAIN_PREFIX := xc32-

ifeq ($(OPTIMIZED_KERNEL_DIR), cmsis_nn)
  CCFLAGS += \
    -I./CMSIS/Core/Include/ \
    -I./CMSIS/DSP/Include/ \
    -I./CMSIS/NN/Include/

  CXXFLAGS += \
    -I./CMSIS/Core/Include/ \
    -I./CMSIS/DSP/Include/ \
    -I./CMSIS/NN/Include/
endif


FLAGS_GCC = -mcpu=$(TARGET_ARCH)


CXXFLAGS += $(FLAGS_GCC)
CCFLAGS += $(FLAGS_GCC)



CCFLAGS += \
 -mthumb \
 -std=c99 \
 -Wall \
 -Wextra \
 -O2 \
 -fdata-sections \
 -ffunction-sections \
 -fomit-frame-pointer \
 -fno-builtin \
 -Wno-unused-parameter \
 -g

CXX_FLAGS += \
 -mthumb \
 -std=c++11 \
 -fno-rtti \
 -fno-exceptions \
 -Wall \
 -Wextra \
 -O3 \
 -fdata-sections \
 -ffunction-sections \
 -fomit-frame-pointer \
 -fno-builtin \
 -Wno-unused-parameter \
 -g

LDFLAGS += \
 -mthumb \
 -u _printf_float \
 -Wl,--gc-sections

LDFLAGS += -lstdc++ \
 -lgcc \
 -lc \
 -lm \
 -lnosys \
