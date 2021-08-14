EXAMPLE_NAME:=micro_speech_lstm
MICRO_SPEECH_LSTM_TEST_SRCS := \
third_party/xtensa/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc

MICRO_SPEECH_LSTM_GENERATOR_INPUTS := \
third_party/xtensa/examples/micro_speech_lstm/micro_speech_lstm.tflite

MICRO_SPEECH_LSTM_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)third_party/xtensa/examples/micro_speech_lstm/micro_speech_lstm_model_data.cc

MICRO_SPEECH_LSTM_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)third_party/xtensa/examples/micro_speech_lstm/micro_speech_lstm_model_data.h \
third_party/xtensa/examples/micro_speech_lstm/no_micro_features_data.h \
third_party/xtensa/examples/micro_speech_lstm/yes_micro_features_data.h \

MICRO_SPEECH_LSTM_HDRS := \
third_party/xtensa/examples/micro_speech_lstm/no_micro_features_data.h \
third_party/xtensa/examples/micro_speech_lstm/yes_micro_features_data.h \

# Tests loading and running a speech model.
$(eval $(call microlite_test,micro_speech_lstm_test,\
$(MICRO_SPEECH_LSTM_TEST_SRCS),,$(MICRO_SPEECH_LSTM_GENERATOR_INPUTS),$(MICRO_SPEECH_LSTM_HDRS)))
