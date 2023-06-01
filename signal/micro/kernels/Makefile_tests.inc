WINDOW_TEST_SRCS := \
$(TENSORFLOW_ROOT)signal/micro/kernels/window_test.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/window_flexbuffers_generated_data.cc

WINDOW_TEST_HDRS := \
$(TENSORFLOW_ROOT)signal/micro/kernels/window_flexbuffers_generated_data.h

$(eval $(call microlite_test,kernel_window_test,\
$(WINDOW_TEST_SRCS),$(WINDOW_TEST_HDRS)))