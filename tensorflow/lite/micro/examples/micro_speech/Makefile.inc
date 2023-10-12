
MICRO_SPEECH_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc

MICRO_SPEECH_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/micro_model_settings.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/micro_test.h

MICRO_SPEECH_SRCS := $(MICRO_SPEECH_TEST_SRCS)

MICRO_SPEECH_HDRS := $(MICRO_SPEECH_TEST_HDRS)

MICRO_SPEECH_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms.wav \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms.wav

MICRO_SPEECH_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.cc

MICRO_SPEECH_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.h

# TODO(b/161489252): Disabling warnings for this example until we have a better
# way to build third_party code with a reduced list of CFLAGS.
# Xtensa hifi5/hifi4 xa_nnlib
CCFLAGS := $(filter-out $(CC_WARNINGS),$(CCFLAGS))

# Tests loading and running a speech model.
$(eval $(call microlite_test,micro_speech_test,\
$(MICRO_SPEECH_TEST_SRCS),$(MICRO_SPEECH_TEST_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,micro_speech,\
$(MICRO_SPEECH_SRCS),$(MICRO_SPEECH_HDRS),$(MICRO_SPEECH_GENERATOR_INPUTS)))

# Add sources and headers generated from $(MICRO_SPEECH_GENERATOR_INPUTS).
MICRO_SPEECH_SRCS += $(MICRO_SPEECH_GENERATED_SRCS)
MICRO_SPEECH_HDRS += $(MICRO_SPEECH_GENERATED_HDRS)

list_micro_speech_example_sources:
	@echo $(MICRO_SPEECH_SRCS)

list_micro_speech_example_headers:
	@echo $(MICRO_SPEECH_HDRS)
