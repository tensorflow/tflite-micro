integration_tests_seanet_pad_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad0.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad1.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad2.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad3.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad4.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad5.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad6.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad7.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad8.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad9.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad10.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad11.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad12.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad13.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad14.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad15.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad16.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad17.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad18.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad0_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad0_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad1_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad1_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad2_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad2_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad3_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad3_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad4_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad4_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad5_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad5_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad6_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad6_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad7_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad7_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad8_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad8_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad9_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad9_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad10_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad10_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad11_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad11_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad12_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad12_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad13_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad13_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad14_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad14_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad15_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad15_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad16_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad16_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad17_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad17_golden_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad18_input0_int16.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/pad18_golden_int16.csv \

integration_tests_seanet_pad_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/seanet/pad/integration_tests.cc  \
$(TENSORFLOW_ROOT)python/tflite_micro/python_ops_resolver.cc \


integration_tests_seanet_pad_HDR := \
$(TENSORFLOW_ROOT)python/tflite_micro/python_ops_resolver.h \


$(eval $(call microlite_test,integration_tests_seanet_pad_test,\
$(integration_tests_seanet_pad_SRCS),$(integration_tests_seanet_pad_HDR),$(integration_tests_seanet_pad_GENERATOR_INPUTS)))