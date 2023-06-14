model_runner_SRCS := \
tensorflow/lite/micro/examples/model_runner/main_functions.cc \
tensorflow/lite/micro/examples/model_runner/model.cc \
tensorflow/lite/micro/examples/model_runner/main.cc

model_runner_HDRS := \
tensorflow/lite/micro/examples/model_runner/main_functions.h \
tensorflow/lite/micro/examples/model_runner/model.h \
tensorflow/lite/micro/examples/model_runner/test_data.h


model_runner_lib_SRCS := \

model_runner_lib_HDRS := \

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/model_runner/*/Makefile.inc)


# Builds a standalone binary
$(eval $(call microlite_test,model_runner,\
$(model_runner_SRCS),$(model_runner_HDRS)))
