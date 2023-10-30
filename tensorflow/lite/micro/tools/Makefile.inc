MICROLITE_TOOL_ROOT_DIR := $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/

LAYER_BY_LAYER_OUPUT_SRCS := \
$(MICROLITE_TOOL_ROOT_DIR)/layer_by_layer.cc \

LAYER_BY_LAYER_OUPUT_HDRS := \
$(MICROLITE_TOOL_ROOT_DIR)benchmarking/op_resolver.h \

ifneq ($(TARGET),bluepill)
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifi5 hifimini))
    $(eval $(call microlite_test,layer_by_layer_output_tool,\
    $(LAYER_BY_LAYER_OUPUT_SRCS),$(LAYER_BY_LAYER_OUPUT_HDRS),))
endif
endif