KEYWORD_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/keyword_benchmark.cc \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.cc

KEYWORD_BENCHMARK_HDRS := \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h \
tensorflow/lite/micro/benchmarks/micro_benchmark.h

PERSON_DETECTION_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/person_detection_benchmark.cc

PERSON_DETECTION_BENCHMARK_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/person_detection/testdata/person.bmp \
tensorflow/lite/micro/examples/person_detection/testdata/no_person.bmp

ifeq ($(CO_PROCESSOR),ethos_u)
  PERSON_DETECTION_BENCHMARK_GENERATOR_INPUTS += \
    tensorflow/lite/micro/models/person_detect.tflite
else
  # Ethos-U use a Vela optimized version of the original model.
  PERSON_DETECTION_BENCHMARK_SRCS += \
  $(GENERATED_SRCS_DIR)tensorflow/lite/micro/models/person_detect_model_data_vela.cc
endif

PERSON_DETECTION_BENCHMARK_HDRS := \
tensorflow/lite/micro/examples/person_detection/model_settings.h \
tensorflow/lite/micro/benchmarks/micro_benchmark.h

# Builds a standalone binary.
$(eval $(call microlite_test,keyword_benchmark,\
$(KEYWORD_BENCHMARK_SRCS),$(KEYWORD_BENCHMARK_HDRS)))

$(eval $(call microlite_test,person_detection_benchmark,\
$(PERSON_DETECTION_BENCHMARK_SRCS),$(PERSON_DETECTION_BENCHMARK_HDRS),$(PERSON_DETECTION_BENCHMARK_GENERATOR_INPUTS)))
