THIRD_PARTY_CC_SRCS += \
  $(wildcard $(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/SigProc/rFFT/asm_src/*.S)

THIRD_PARTY_CC_HDRS += \
  $(wildcard $(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/include/*.h)

INCLUDES += \
  -I$(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/include \
  -I$(HEXAGON_ROOT)/$(HEXAGON_TOOL_VER)/Examples/libcore/SigProc/rFFT/include
