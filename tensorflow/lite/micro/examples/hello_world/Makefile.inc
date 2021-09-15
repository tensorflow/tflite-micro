EXAMPLE_NAME:=hello_world

HELLO_WORLD_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h

HELLO_WORLD_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.cc

HELLO_WORLD_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.h

HELLO_WORLD_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/hello_world/hello_world.tflite

HELLO_WORLD_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/hello_world/hello_world_model_data.cc

HELLO_WORLD_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/hello_world/hello_world_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(HELLO_WORLD_TEST_SRCS),,$(HELLO_WORLD_GENERATOR_INPUTS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,$(EXAMPLE_NAME),\
$(HELLO_WORLD_SRCS),$(HELLO_WORLD_HDRS),$(HELLO_WORLD_GENERATOR_INPUTS)))

# Add sources and headers generated from $(HELLO_WORLD_GENERATOR_INPUTS).
HELLO_WORLD_SRCS += $(HELLO_WORLD_GENERATED_SRCS)
HELLO_WORLD_HDRS += $(HELLO_WORLD_GENERATED_HDRS)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(HELLO_WORLD_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(HELLO_WORLD_HDRS)
