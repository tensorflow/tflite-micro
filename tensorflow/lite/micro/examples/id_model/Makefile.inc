EXAMPLE_NAME:=id_model

ID_MODEL_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

ID_MODEL_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_utils.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/settings.h

ID_MODEL_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME).tflite

ID_MODEL_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.cc

ID_MODEL_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.h

# Tests loading and running the model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(ID_MODEL_TEST_SRCS),,$(ID_MODEL_GENERATOR_INPUTS)))

# Add sources and headers generated from $(ID_MODEL_GENERATOR_INPUTS).
_SRCS += $(ID_MODEL_GENERATED_SRCS)
_HDRS += $(ID_MODEL_GENERATED_HDRS)

# Compier flags for OpenCV libraries and header files.
INCLUDES += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(ID_MODEL_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(ID_MODEL_HDRS)
