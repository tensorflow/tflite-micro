EXAMPLE_NAME:=add_model_float

ADD_MODEL_FLOAT_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h

ADD_MODEL_FLOAT_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.cc

ADD_MODEL_FLOAT_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.h

ADD_MODEL_FLOAT_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/add_model_float/add_model_float.tflite

ADD_MODEL_FLOAT_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/add_model_float/add_model_float_model_data.cc

ADD_MODEL_FLOAT_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/add_model_float/add_model_float_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(ADD_MODEL_FLOAT_TEST_SRCS),,$(ADD_MODEL_FLOAT_GENERATOR_INPUTS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,$(EXAMPLE_NAME),\
$(ADD_MODEL_FLOAT_SRCS),$(ADD_MODEL_FLOAT_HDRS),$(ADD_MODEL_FLOAT_GENERATOR_INPUTS)))

# Add sources and headers generated from $(ADD_MODEL_FLOAT_GENERATOR_INPUTS).
ADD_MODEL_FLOAT_SRCS += $(ADD_MODEL_FLOAT_GENERATED_SRCS)
ADD_MODEL_FLOAT_HDRS += $(ADD_MODEL_FLOAT_GENERATED_HDRS)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(ADD_MODEL_FLOAT_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(ADD_MODEL_FLOAT_HDRS)
