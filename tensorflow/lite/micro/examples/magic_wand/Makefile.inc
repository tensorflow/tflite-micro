ACCELEROMETER_HANDLER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/accelerometer_handler_test.cc

ACCELEROMETER_HANDLER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h

OUTPUT_HANDLER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/output_handler.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/output_handler_test.cc

OUTPUT_HANDLER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/output_handler.h

GESTURE_PREDICTOR_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/gesture_predictor.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/gesture_predictor_test.cc

GESTURE_PREDICTOR_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/constants.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h \

magic_wand_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/magic_wand_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/slope_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/ring_micro_features_data.cc

magic_wand_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/slope_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/ring_micro_features_data.h

magic_wand_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/main_functions.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/gesture_predictor.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/output_handler.cc

magic_wand_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/main_functions.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/constants.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/output_handler.h

magic_wand_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/magic_wand.tflite

magic_wand_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.cc

magic_wand_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/magic_wand/*/Makefile.inc)

# Tests the accelerometer handler
$(eval $(call microlite_test,gesture_accelerometer_handler_test,\
$(ACCELEROMETER_HANDLER_TEST_SRCS),$(ACCELEROMETER_HANDLER_TEST_HDRS)))

# Tests the output handler
$(eval $(call microlite_test,gesture_output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Tests the gesture predictor
$(eval $(call microlite_test,gesture_predictor_test,\
$(GESTURE_PREDICTOR_TEST_SRCS),$(GESTURE_PREDICTOR_TEST_HDRS)))

# Tests loading and running the gesture recognition model
$(eval $(call microlite_test,magic_wand_test,\
$(magic_wand_TEST_SRCS),$(magic_wand_TEST_HDRS),$(magic_wand_GENERATOR_INPUTS)))

# Builds a standalone binary
$(eval $(call microlite_test,magic_wand,\
$(magic_wand_SRCS),$(magic_wand_HDRS),$(magic_wand_GENERATOR_INPUTS)))

magic_wand_SRCS += $(magic_wand_GENERATED_SRCS)
magic_wand_HDRS += $(magic_wand_GENERATED_HDRS)

list_magic_wand_example_sources:
	@echo $(magic_wand_SRCS)

list_magic_wand_example_headers:
	@echo $(magic_wand_HDRS)