EXAMPLE_NAME:=dp_net

DPNET_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h

DPNET_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.cc

DPNET_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.h

DPNET_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/dp_net/dp_net.tflite

DPNET_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/dp_net/dp_net_model_data.cc

DPNET_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/dp_net/dp_net_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(DPNET_TEST_SRCS),,$(DPNET_GENERATOR_INPUTS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,$(EXAMPLE_NAME),\
$(DPNET_SRCS),$(DPNET_HDRS),$(DPNET_GENERATOR_INPUTS)))

# Add sources and headers generated from $(DPNET_GENERATOR_INPUTS).
DPNET_SRCS += $(DPNET_GENERATED_SRCS)
DPNET_HDRS += $(DPNET_GENERATED_HDRS)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(DPNET_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(DPNET_HDRS)
