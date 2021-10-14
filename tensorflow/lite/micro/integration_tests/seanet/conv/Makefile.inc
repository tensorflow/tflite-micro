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
tensorflow/lite/micro/integration_tests/seanet/conv/conv0_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv0_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv1_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv1_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv2_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv2_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv3_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv3_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv4_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv4_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv5_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv5_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv6_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv6_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv7_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv7_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv8_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv8_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv9_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv9_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv10_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv10_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv11_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv11_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv12_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv12_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv13_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv13_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv14_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv14_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv15_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv15_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv16_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv16_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv17_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv17_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv18_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv18_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv19_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv19_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv20_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv20_golden.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv21_input.csv \
tensorflow/lite/micro/integration_tests/seanet/conv/conv21_golden.csv \

integration_tests_seanet_conv_SRCS := \
tensorflow/lite/micro/integration_tests/seanet/conv/integration_tests.cc

$(eval $(call microlite_test,integration_tests_seanet_conv_test,\
$(integration_tests_seanet_conv_SRCS),,$(integration_tests_seanet_conv_GENERATOR_INPUTS)))
