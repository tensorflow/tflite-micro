
DTLN_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/dtln_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/dtln_inout_data.cc

DTLN_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/dtln_inout_data.h

DTLN_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite

DTLN_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/dtln_noise_suppression_model_data.cc

DTLN_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/dtln_noise_suppression_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/dtln/*/Makefile.inc)

# TODO(b/161489252): Disabling warnings for this example until we have a better
# way to build third_party code with a reduced list of CFLAGS.
CCFLAGS := $(filter-out $(CC_WARNINGS),$(CCFLAGS))

# Tests loading and running a dtln model.
# TODO(b/319712246): Re-enable tests once the dtln_test works on HiFi Mini & VP6
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini vision_p6))
  $(eval $(call microlite_test,dtln_test,\
  $(DTLN_TEST_SRCS),$(DTLN_TEST_HDRS),$(DTLN_GENERATOR_INPUTS)))
endif
