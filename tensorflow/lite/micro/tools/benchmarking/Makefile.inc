MICROLITE_BENCHMARK_ROOT_DIR := tensorflow/lite/micro/tools/benchmarking

GENERIC_BENCHMARK_SRCS := $(MICROLITE_BENCHMARK_ROOT_DIR)/internal/generic_model_benchmark.cc \
$(MICROLITE_BENCHMARK_ROOT_DIR)/internal/log_utils.cc \
$(MICROLITE_BENCHMARK_ROOT_DIR)/internal/metrics.cc \
# tensorflow/lite/tools/make/downloads/flatbuffers/src/util.cpp

GENERIC_BENCHMARK_HDRS := $(MICROLITE_BENCHMARK_ROOT_DIR)/op_resolver.h \
$(MICROLITE_BENCHMARK_ROOT_DIR)/internal/log_utils.h \
$(MICROLITE_BENCHMARK_ROOT_DIR)/internal/metrics.h \
$(MICROLITE_BENCHMARK_ROOT_DIR)/internal/micro_benchmark.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/util.h

ifdef INPUT_DATA_TYPE
  CXXFLAGS += -DINPUT_DATA_TYPE=$(INPUT_DATA_TYPE)
endif

THIRD_PARTY_CC_SRCS += $(DOWNLOADS_DIR)/flatbuffers/src/util.cpp

ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifi5 hifimini))
  $(eval $(call microlite_test,tflm_benchmark,\
  $(GENERIC_BENCHMARK_SRCS),$(GENERIC_BENCHMARK_HDRS),))
endif