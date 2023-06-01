MICROLITE_CC_KERNEL_SRCS += \
$(TENSORFLOW_ROOT)signal/micro/kernels/window.cc \
$(TENSORFLOW_ROOT)signal/src/window.cc

MICROLITE_CC_HDRS += \
$(wildcard $(TENSORFLOW_ROOT)signal/micro/kernels/*.h) \
$(wildcard $(TENSORFLOW_ROOT)signal/src/*.h)
