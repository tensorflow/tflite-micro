HELLO_WORLD_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/hello_world_test.cc

HELLO_WORLD_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/hello_world_test.cc

HELLO_WORLD_HDRS :=

HELLO_WORLD_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_float.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite

HELLO_WORLD_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_int8_model_data.cc

HELLO_WORLD_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_int8_model_data.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,hello_world_test,\
$(HELLO_WORLD_TEST_SRCS),,$(HELLO_WORLD_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,hello_world,\
$(HELLO_WORLD_SRCS),,$(HELLO_WORLD_GENERATOR_INPUTS)))

# Add sources and headers generated from $(HELLO_WORLD_GENERATOR_INPUTS).
HELLO_WORLD_SRCS += $(HELLO_WORLD_GENERATED_SRCS)
HELLO_WORLD_HDRS += $(HELLO_WORLD_GENERATED_HDRS)

list_hello_world_example_sources:
	@echo $(HELLO_WORLD_SRCS)

list_hello_world_example_headers:
	@echo $(HELLO_WORLD_HDRS)
