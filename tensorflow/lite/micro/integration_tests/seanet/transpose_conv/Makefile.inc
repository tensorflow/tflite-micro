integration_tests_seanet_transpose_conv_GENERATOR_INPUTS := \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0.tflite \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1.tflite \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2.tflite \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3.tflite \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4.tflite \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_input0_int32.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_input0_int32.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_input0_int32.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_input0_int32.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_input0_int32.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_golden_int16.csv \

integration_tests_seanet_transpose_conv_SRCS := \
tensorflow/lite/micro/integration_tests/seanet/transpose_conv/integration_tests.cc

$(eval $(call microlite_test,integration_tests_seanet_transpose_conv_test,\
$(integration_tests_seanet_transpose_conv_SRCS),,$(integration_tests_seanet_transpose_conv_GENERATOR_INPUTS)))