EXAMPLE_NAME:=retinaface_torch

RETINAFACE_TORCH_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

RETINAFACE_TORCH_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_utils.h

RETINAFACE_TORCH_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME).tflite

RETINAFACE_TORCH_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.cc

RETINAFACE_TORCH_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.h

# Tests loading and running the model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(RETINAFACE_TORCH_TEST_SRCS),,$(RETINAFACE_TORCH_GENERATOR_INPUTS)))

# Add sources and headers generated from $(RETINAFACE_TORCH_GENERATOR_INPUTS).
RETINAFACE_TORCH_SRCS += $(RETINAFACE_TORCH_GENERATED_SRCS)
RETINAFACE_TORCH_HDRS += $(RETINAFACE_TORCH_GENERATED_HDRS)

# Compier flags for OpenCV libraries and header files.
INCLUDES += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(RETINAFACE_TORCH_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(RETINAFACE_TORCH_HDRS)
