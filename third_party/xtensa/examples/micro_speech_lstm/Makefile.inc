# The lstm kernel is currently only implemented for OPTIMIZED_KERNEL_DIR=xtensa
# and as a result, this partcular example is also excluded in all other cases.


# TODO(#715): Adding support for nullptr bias in fully connected resulted in the
# micro_speech_lstm example crashing on invoke to LSTM. This is somewhat
# unexpected and the test is being disabled so that the root-cause can be
# investigated independent of landing support for nullptr bias in fully_connected
# for all the platforms.
#
ifeq ($(OPTIMIZED_KERNEL_DIR), xtensa)
   EXAMPLE_NAME:=micro_speech_lstm
   MICRO_SPEECH_LSTM_TEST_SRCS := \
   third_party/xtensa/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc \
   third_party/xtensa/examples/micro_speech_lstm/no_micro_features_data.cc \
   third_party/xtensa/examples/micro_speech_lstm/yes_micro_features_data.cc

   MICRO_SPEECH_LSTM_GENERATOR_INPUTS := \
   third_party/xtensa/examples/micro_speech_lstm/micro_speech_lstm.tflite

   MICRO_SPEECH_LSTM_HDRS := \
   third_party/xtensa/examples/micro_speech_lstm/no_micro_features_data.h \
   third_party/xtensa/examples/micro_speech_lstm/yes_micro_features_data.h

   # TODO(#802): Disbaling test that takes a long time to run.
   ## Tests loading and running a speech model.
   #$(eval $(call microlite_test,micro_speech_lstm_test,\
   #$(MICRO_SPEECH_LSTM_TEST_SRCS),$(MICRO_SPEECH_LSTM_HDRS),$(MICRO_SPEECH_LSTM_GENERATOR_INPUTS)))
endif
