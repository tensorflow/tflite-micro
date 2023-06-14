TARGET_DEFAULT_TOOLCHAIN_ROOT := ${TENSORFLOW_ROOT}$(MAKEFILE_DIR)/downloads/gcc_embedded/bin/
TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

GCC_TARGET_ARCH := $(TARGET_ARCH)
CORE=M33
ARM_LDFLAGS := -Wl,--cpu=Cortex-M33
FLOAT=hard


GECKO_SDK_PATH=$(GECKO_SDK_PATH)
CMSIS_PATH=$(CMSIS_PATH)

SILABS_CFLAGS = '-DNDEBUG=1' \
 '-DTF_LITE_STATIC_MEMORY=1' \
 '-DEFR32MG24B310F1536IM48=1' \
 '-DSL_BOARD_NAME="BRD4001A"' \
 '-DSL_BOARD_REV="A03"' \
 '-DSL_COMPONENT_CATALOG_PRESENT=1' \
 '-DTF_LITE_MCU_DEBUG_LOG=1' \
 '-DCMSIS_NN=1'


FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH) -mfpu=auto


CXXFLAGS += $(FLAGS_GCC)
CCFLAGS += $(FLAGS_GCC)

CCFLAGS +=  $(SILABS_CFLAGS)
CXXFLAGS +=  $(SILABS_CFLAGS)



CCFLAGS += \
 -mcpu=cortex-m33 \
 -mthumb \
 -mfpu=fpv5-sp-d16 \
 -mfloat-abi=hard \
 -std=c99 \
 -Wall \
 -Wextra \
 -O3 \
 -fdata-sections \
 -ffunction-sections \
 -fomit-frame-pointer \
 -fno-builtin \
 -imacros sl_gcc_preinclude.h \
 -Wno-unused-parameter \
 -mfp16-format=ieee \
 --specs=nano.specs \
 -g

CXX_FLAGS += \
 -mcpu=cortex-m33 \
 -mthumb \
 -mfpu=fpv5-sp-d16 \
 -mfloat-abi=hard \
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
 -mfp16-format=ieee \
 -g

LDFLAGS += \
 -mcpu=cortex-m33 \
 -mthumb \
 -mfpu=fpv5-sp-d16 \
 -mfloat-abi=hard \
 -u _printf_float \
 -Wl,--gc-sections

LDFLAGS += -lstdc++ \
 -lgcc \
 -lc \
 -lm \
 -lnosys \
 -mfpu=fpv5-sp-d16 
