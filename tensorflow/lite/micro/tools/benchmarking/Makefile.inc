MICROLITE_BENCHMARK_ROOT_DIR := $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/benchmarking

GENERIC_BENCHMARK_SRCS := \
$(MICROLITE_BENCHMARK_ROOT_DIR)/generic_model_benchmark.cc \
$(MICROLITE_BENCHMARK_ROOT_DIR)/log_utils.cc \
$(MICROLITE_BENCHMARK_ROOT_DIR)/metrics.cc

GENERIC_BENCHMARK_HDRS := \
$(MICROLITE_BENCHMARK_ROOT_DIR)/op_resolver.h \
$(MICROLITE_BENCHMARK_ROOT_DIR)/log_utils.h \
$(MICROLITE_BENCHMARK_ROOT_DIR)/metrics.h

ifneq ($(TARGET),bluepill)
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifi5 hifimini))
    $(eval $(call microlite_test,tflm_benchmark,\
    $(GENERIC_BENCHMARK_SRCS),$(GENERIC_BENCHMARK_HDRS),))
endif
endif