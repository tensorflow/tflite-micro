integration_tests_seanet_conv_GENERATOR_INPUTS := \
tensorflow/lite/micro/integration_tests/seanet/conv/conv0.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv1.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv2.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv3.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv4.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv5.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv6.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv7.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv8.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv9.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv10.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv11.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv12.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv13.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv14.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv15.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv16.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv17.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv18.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv19.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv20.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv21.tflite \
tensorflow/lite/micro/integration_tests/seanet/conv/conv0_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv0_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv1_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv1_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv2_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv2_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv3_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv3_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv4_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv4_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv5_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv5_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv6_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv6_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv7_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv7_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv8_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv8_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv9_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv9_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv10_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv10_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv11_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv11_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv12_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv12_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv13_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv13_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv14_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv14_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv15_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv15_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv16_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv16_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv17_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv17_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv18_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv18_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv19_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv19_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv20_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv20_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv21_input_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv21_golden_int16.csv \

integration_tests_seanet_conv_SRCS := \
tensorflow/lite/micro/integration_tests/seanet/conv/integration_tests.cc

$(eval $(call microlite_test,integration_tests_seanet_conv,\
$(integration_tests_seanet_conv_SRCS),,$(integration_tests_seanet_conv_GENERATOR_INPUTS)))
