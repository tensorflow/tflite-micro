TARGET_ARCH :=

ifeq ($(TARGET_ARCH), )
  $(error TARGET_ARCH must be specified on the command line)
endif

# Create a cflag based on the specified TARGET_ARCH. For example:
#   TARGET_ARCH=CEVA_BX1 --> -DCEVA_BX1
#   TARGET_ARCH=CEVA_SP500 --> -DCEVA_SP500
TARGET_ARCH_DEFINES := -D$(shell echo $(TARGET_ARCH) | tr [a-z] [A-Z])

CORE_OPTIMIZATION_LEVEL := -O4

#no need for -lm:
MICROLITE_LIBS :=

TARGET_TOOLCHAIN_PREFIX := 
CXX_TOOL = clang++
CC_TOOL = clang
LD_TOOL = ceva-elf-ld
LD = ceva-elf-ld
AR = ceva-elf-ar

PLATFORM_FLAGS = \
$(TARGET_ARCH_DEFINES) \
	-fmessage-length=0 \
	-fpermissive \
	-O4 \
	-g3 \
	-Wall \
	-pedantic \
	-D_LIBCPP_INLINE_VISIBILITY="" \
	-D_LIBCPP_EXTERN_TEMPLATE_INLINE_VISIBILITY=""
	 

CXXFLAGS := -std=c++17 -DTF_LITE_STATIC_MEMORY 
CCFLAGS  := -std=c17   -DTF_LITE_STATIC_MEMORY 

ifeq ($(TARGET_ARCH), CEVA_BX1)
PLATFORM_FLAGS += \
	--target=cevabx1-elf \
	-mcpu=cevabx1v1.0.0 \
	-m32x32 \
	-mgetbits \
	-mloop-buffer-size=10 \
	-mfp=1 \
	-mdpfp=1


LDFLAGS += \
	  -T \
	tensorflow/lite/micro/tools/make/targets/ceva/CEVA_BX1_TFLM_18.0.5.ld \
     -L${TARGET_TOOLCHAIN_ROOT}_cevatools/lib/clang/9.0.1/cevabx1-unknown-unknown-elf/rtlv1.0.0-fp1-dpfp1/lib/ \
    -lc++ -lc++abi -lc -lcompiler-rt -lCEVA_TFLM_lib -lceva_dsp_lib  \
	--for-linker --no-relax \
	--for-linker --no-gc-sections \
	--for-linker -defsym \
	--for-linker __internal_data_size=2048k \
	--for-linker -defsym \
	--for-linker __internal_code_size=256k \

endif

ifeq ($(TARGET_ARCH), CEVA_SP500)
PLATFORM_FLAGS = \
 -pedantic \
 -Wa,--no-rstr-check \
 --target=senspro-elf \
 -mcpu=sensprov1.0.0 \
 -mvu=1 \
 -mno-vld2 \
 -mvmpyv5 \
 -mvmpyext -mnonlinear=1 -mno-vbnn -mvhist \
 -mlvu=1 \
 -mfp=2 \
 -mdpfp=2 \
 -mvfp=1

  LDFLAGS += \
--no-relax --no-gc-sections \
 -defsym __internal_code_size=0k \
 -defsym __internal_data_size=512k 
 
endif

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

MICROLITE_CC_HDRS += \
  tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h \
  tensorflow/lite/micro/kernels/ceva/types.h \
  tensorflow/lite/micro/kernels/ceva/ceva_common.h 
  

MICROLITE_CC_SRCS += \
  tensorflow/lite/micro/kernels/ceva/ceva_common.cc
