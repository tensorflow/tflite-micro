# TODO(b/288938993): de-dupe internal wrapper directory and remove this
MICROLITE_CC_KERNEL_SRCS += \
$(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_float.cc \
$(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_int16.cc \
$(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_int32.cc

MICROLITE_CC_HDRS += \
$(wildcard $(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/*.h) \
