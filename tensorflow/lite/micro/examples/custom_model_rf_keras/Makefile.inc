EXAMPLE_NAME:=custom_model_rf_keras

CUSTOM_MODEL_RF_KERAS_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

CUSTOM_MODEL_RF_KERAS_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_utils.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/settings.h

CUSTOM_MODEL_RF_KERAS_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME).tflite

CUSTOM_MODEL_RF_KERAS_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.cc

CUSTOM_MODEL_RF_KERAS_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.h

# Tests loading and running the model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(CUSTOM_MODEL_RF_KERAS_TEST_SRCS),,$(CUSTOM_MODEL_RF_KERAS_GENERATOR_INPUTS)))

# Add sources and headers generated from $(CUSTOM_MODEL_RF_KERAS_GENERATOR_INPUTS).
_SRCS += $(CUSTOM_MODEL_RF_KERAS_GENERATED_SRCS)
_HDRS += $(CUSTOM_MODEL_RF_KERAS_GENERATED_HDRS)

# Compier flags for OpenCV libraries and header files.
INCLUDES += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(CUSTOM_MODEL_RF_KERAS_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(CUSTOM_MODEL_RF_KERAS_HDRS)
