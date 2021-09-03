MICROLITE_CC_KERNEL_SRCS += \
tensorflow/lite/micro/kernels/hexagon/fully_connected_int8.cc \
tensorflow/lite/micro/kernels/hexagon/svdf_int8.cc

# Full path to the hexagon_tflm static library.
HEXAGON_TFLM_LIB :=

ifeq ($(HEXAGON_TFLM_LIB), )
  $(error HEXAGON_TFLM_LIB must be specified on the command line)
endif

MICROLITE_LIBS += $(HEXAGON_TFLM_LIB)

THIRD_PARTY_CC_SRCS += \
  $(wildcard $(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/SigProc/rFFT/asm_src/*.S)

THIRD_PARTY_CC_HDRS += \
  $(wildcard $(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/include/*.h)

INCLUDES += \
  -I$(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/include \
  -I$(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/SigProc/rFFT/include
