integration_tests_seanet_add_GENERATOR_INPUTS := \
tensorflow/lite/micro/integration_tests/seanet/add/add0.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add1.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add2.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add3.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add4.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add5.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add6.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add7.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add8.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add9.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add10.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add11.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add12.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add13.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add14.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add15.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add16.tflite \
tensorflow/lite/micro/integration_tests/seanet/add/add0_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add0_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add0_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add1_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add1_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add1_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add2_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add2_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add2_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add3_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add3_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add3_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add4_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add4_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add4_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add5_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add5_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add5_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add6_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add6_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add6_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add7_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add7_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add7_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add8_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add8_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add8_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add9_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add9_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add9_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add10_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add10_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add10_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add11_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add11_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add11_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add12_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add12_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add12_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add13_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add13_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add13_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add14_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add14_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add14_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add15_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add15_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add15_golden_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add16_input0_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add16_input1_int16.csv \
tensorflow/lite/micro/integration_tests/seanet/add/add16_golden_int16.csv \

integration_tests_seanet_add_SRCS := \
tensorflow/lite/micro/integration_tests/seanet/add/integration_tests.cc

$(eval $(call microlite_test,integration_tests_seanet_add_test,\
$(integration_tests_seanet_add_SRCS),,$(integration_tests_seanet_add_GENERATOR_INPUTS)))