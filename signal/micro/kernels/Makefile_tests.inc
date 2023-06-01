
WINDOW_TEST_SRCS := \
$(TENSORFLOW_ROOT)signal/micro/kernels/window_test.cc

FILTER_BANK_TEST_HDRS := \
$(TENSORFLOW_ROOT)signal/micro/kernels/register_signal_ops.h

$(eval $(call microlite_test,kernel_window_test,\
$(WINDOW_TEST_SRCS),$(WINDOW_TEST_HDRS)))
