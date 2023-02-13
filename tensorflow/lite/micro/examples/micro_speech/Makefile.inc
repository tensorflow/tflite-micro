
INCLUDES += \
 -I$(MAKEFILE_DIR)/downloads/kissfft

KISSFFT_LIB_SRCS :=

KISSFFT_LIB_HDRS := \
$(MAKEFILE_DIR)/downloads/kissfft/COPYING \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.c \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.h \
$(MAKEFILE_DIR)/downloads/kissfft/_kiss_fft_guts.h \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.c \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.h

MICRO_SPEECH_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc

MICRO_SPEECH_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \

SIMPLE_FEATURES_GENERATOR_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/no_power_spectrum_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/yes_power_spectrum_data.cc

SIMPLE_FEATURES_GENERATOR_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/simple_model_settings.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/no_power_spectrum_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/simple_features/yes_power_spectrum_data.h

MICRO_FEATURES_LIB_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft_util.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_lut.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window_util.c \
$(KISSFFT_LIB_SRCS)

MICRO_FEATURES_LIB_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/bits.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/kiss_fft_common.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_lut.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window_util.h \
$(KISSFFT_LIB_HDRS)

MICRO_FEATURES_GENERATOR_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
$(MICRO_FEATURES_LIB_SRCS)

MICRO_FEATURES_GENERATOR_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h \
$(MICRO_FEATURES_LIB_HDRS)

MICRO_FEATURES_GENERATOR_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_feature_data_slice.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_feature_data_slice.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

MICRO_FEATURES_GENERATOR_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_feature_data_slice.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_feature_data_slice.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

AUDIO_PROVIDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.cc

AUDIO_PROVIDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.h \

AUDIO_PROVIDER_MOCK_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider_mock_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider_mock.cc

AUDIO_PROVIDER_MOCK_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.h \

FEATURE_PROVIDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

FEATURE_PROVIDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

FEATURE_PROVIDER_MOCK_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider_mock.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

FEATURE_PROVIDER_MOCK_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

RECOGNIZE_COMMANDS_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands.cc

RECOGNIZE_COMMANDS_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands.h

COMMAND_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder.cc

COMMAND_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder.h

MICRO_SPEECH_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/main_functions.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

MICRO_SPEECH_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/main_functions.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

MICRO_SPEECH_MOCK_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/main_functions.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider_mock.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

MICRO_SPEECH_MOCK_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/audio_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/feature_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/recognize_commands.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/command_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/main_functions.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

MICRO_SPEECH_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms.wav

MICRO_SPEECH_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_speech_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.cc

MICRO_SPEECH_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_speech_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/*/Makefile.inc)

# TODO(b/161489252): Disabling warnings for this example until we have a better
# way to build third_party code with a reduced list of CFLAGS.
CCFLAGS := $(filter-out $(CC_WARNINGS),$(CCFLAGS))

# Test the code for feature generation.
$(eval $(call microlite_test,micro_features_generator_test,\
$(MICRO_FEATURES_GENERATOR_TEST_SRCS),$(MICRO_FEATURES_GENERATOR_TEST_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Tests loading and running a speech model.
$(eval $(call microlite_test,micro_speech_test,\
$(MICRO_SPEECH_TEST_SRCS),$(MICRO_SPEECH_TEST_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), hifimini)
  # Test the code for feature generation.
  $(eval $(call microlite_test,simple_features_generator_test,\
  $(SIMPLE_FEATURES_GENERATOR_TEST_SRCS),$(SIMPLE_FEATURES_GENERATOR_TEST_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))
endif

# Tests the audio provider module.
$(eval $(call microlite_test,audio_provider_test,\
$(AUDIO_PROVIDER_TEST_SRCS),$(AUDIO_PROVIDER_TEST_HDRS)))

# Tests the audio provider mock module.
$(eval $(call microlite_test,audio_provider_mock_test,\
$(AUDIO_PROVIDER_MOCK_TEST_SRCS),$(AUDIO_PROVIDER_MOCK_TEST_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Tests the feature provider module.
$(eval $(call microlite_test,feature_provider_test,\
$(FEATURE_PROVIDER_TEST_SRCS),$(FEATURE_PROVIDER_TEST_HDRS)))

# Tests the feature provider module using the mock audio provider.
$(eval $(call microlite_test,feature_provider_mock_test,\
$(FEATURE_PROVIDER_MOCK_TEST_SRCS),$(FEATURE_PROVIDER_MOCK_TEST_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Tests the command recognizer module.
$(eval $(call microlite_test,recognize_commands_test,\
$(RECOGNIZE_COMMANDS_TEST_SRCS),$(RECOGNIZE_COMMANDS_TEST_HDRS)))

# Tests responding to a command.
$(eval $(call microlite_test,command_responder_test,\
$(COMMAND_RESPONDER_TEST_SRCS),$(COMMAND_RESPONDER_TEST_HDRS)))

# Builds a standalone speech command recognizer binary.
$(eval $(call microlite_test,micro_speech,\
$(MICRO_SPEECH_SRCS),$(MICRO_SPEECH_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Builds a standalone speech command recognizer binary using fake audio input.
$(eval $(call microlite_test,micro_speech_mock,\
$(MICRO_SPEECH_MOCK_SRCS),$(MICRO_SPEECH_MOCK_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Add sources and headers generated from $(MICRO_SPEECH_GENERATOR_INPUTS).
MICRO_SPEECH_SRCS += $(MICRO_SPEECH_GENERATED_SRCS)
MICRO_SPEECH_HDRS += $(MICRO_SPEECH_GENERATED_HDRS)

MICRO_SPEECH_MOCK_SRCS += $(MICRO_SPEECH_GENERATED_SRCS)
MICRO_SPEECH_MOCK_HDRS += $(MICRO_SPEECH_GENERATED_HDRS)

list_micro_speech_example_sources:
	@echo $(MICRO_SPEECH_SRCS)

list_micro_speech_example_headers:
	@echo $(MICRO_SPEECH_HDRS)

list_micro_speech_mock_example_sources:
	@echo $(MICRO_SPEECH_MOCK_SRCS)

list_micro_speech_mock_example_headers:
	@echo $(MICRO_SPEECH_MOCK_HDRS)
