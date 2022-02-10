integration_tests_seanet_quantize_GENERATOR_INPUTS := \
tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0.tflite \
tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1.tflite \
tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_input0_int32.csv \
tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_golden_int32.csv

integration_tests_seanet_quantize_SRCS := \
tensorflow/lite/micro/integration_tests/seanet/quantize/integration_tests.cc

$(eval $(call microlite_test,integration_tests_seanet_quantize_test,\
$(integration_tests_seanet_quantize_SRCS),,$(integration_tests_seanet_quantize_GENERATOR_INPUTS)))
