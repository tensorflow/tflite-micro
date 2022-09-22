integration_tests_seanet_quantize_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_input0_int32.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_golden_int32.csv

integration_tests_seanet_quantize_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/quantize/integration_tests.cc

$(eval $(call microlite_test,integration_tests_seanet_quantize_test,\
$(integration_tests_seanet_quantize_SRCS),,$(integration_tests_seanet_quantize_GENERATOR_INPUTS)))
