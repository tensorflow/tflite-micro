EXAMPLE_NAME:=retinaface

RETINAFACE_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

RETINAFACE_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_utils.h

RETINAFACE_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME).tflite

RETINAFACE_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.cc

RETINAFACE_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_model_data.h

# Tests loading and running the model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(RETINAFACE_TEST_SRCS),,$(RETINAFACE_GENERATOR_INPUTS)))

# Add sources and headers generated from $(RETINAFACE_GENERATOR_INPUTS).
RETINAFACE_SRCS += $(RETINAFACE_GENERATED_SRCS)
RETINAFACE_HDRS += $(RETINAFACE_GENERATED_HDRS)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(RETINAFACE_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(RETINAFACE_HDRS)
