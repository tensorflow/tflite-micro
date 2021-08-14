MICRO_SPEECH_TEST_SRCS := \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech_lstm/micro_speech_lstm_test.cc \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech_lstm/model.cc \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech_lstm/no_micro_features_data.cc \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech_lstm/yes_micro_features_data.cc

MICRO_SPEECH_TEST_HDRS := \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech/model.h \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech/no_micro_features_data.h \
tensorflow/lite/micro/kernels/xtensa/lstm/examples/micro_speech/yes_micro_features_data.h \

# Tests loading and running a speech model.
$(eval $(call microlite_test,micro_speech_lstm_test,\
$(MICRO_SPEECH_TEST_SRCS),$(MICRO_SPEECH_TEST_HDRS)))
