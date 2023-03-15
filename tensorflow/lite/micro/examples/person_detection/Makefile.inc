person_detection_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/model_settings.cc

person_detection_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/model_settings.h

person_detection_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/person_detection_test.cc \
$(person_detection_MODEL_SRCS)

person_detection_TEST_HDRS := \
$(person_detection_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/image_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/model_settings.h

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/detection_responder.h

person_detection_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/main_functions.cc \
$(person_detection_MODEL_SRCS)

person_detection_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/main_functions.h \
$(person_detection_MODEL_HDRS)

person_detection_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/person_detect.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/testdata/person.bmp \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/testdata/no_person.bmp

person_detection_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/person_detect_model_data.cc

person_detection_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/person_detect_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,person_detection_test,\
  $(person_detection_TEST_SRCS),$(person_detection_TEST_HDRS),$(person_detection_GENERATOR_INPUTS)))
endif

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,person_detection,\
$(person_detection_SRCS),$(person_detection_HDRS),$(person_detection_GENERATOR_INPUTS)))

# Add sources and headers generated from $(person_detection_GENERATOR_INPUTS).
person_detection_SRCS += $(person_detection_GENERATED_SRCS)
person_detection_HDRS += $(person_detection_GENERATED_HDRS)

list_person_detection_example_sources:
	@echo $(person_detection_SRCS)

list_person_detection_example_headers:
	@echo $(person_detection_HDRS)
