
# Explicitly add kernel sources specific to the Xtensa optimized
# implementations.
MICROLITE_CC_KERNEL_SRCS += \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/add_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_common_xtensa.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_hifi.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_int16_reference.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_int8_int16.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_int8_reference.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/depthwise_conv_hifi.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/depthwise_conv_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/fully_connected_common_xtensa.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/fully_connected_int8.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/fully_connected_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/pad_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/pooling_int8.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/pooling_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/reduce_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/reshape_vision.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/softmax_int8_int16.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/softmax_vision.cc

ifeq ($(TARGET_ARCH), hifimini)
  # hifimini optimizations are implemented in the TFLM repository itself.
  THIRD_PARTY_KERNEL_CC_SRCS += \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/hifimini/svdf.cc \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/hifimini/fully_connected.cc

  FFT_PATH := $(MAKEFILE_DIR)/downloads/hifi_fft
  INCLUDES += -I$(FFT_PATH)/

  THIRD_PARTY_KERNEL_CC_SRCS += \
    $(shell find $(FFT_PATH)/hifi2_fft -name "*.c")
  THIRD_PARTY_CC_HDRS += \
    $(shell find $(FFT_PATH)/hifi2_fft -name "*.h")

else ifeq ($(TARGET_ARCH), hifi5)
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${DOWNLOADS_DIR} hifi5 $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_ndsp_download.sh ${DOWNLOADS_DIR} hifi5 $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa ndsp download: $(DOWNLOAD_RESULT))
  endif

  # TODO(b/161489252): -Wno-shadow is only needed for xannlib. But since we do
  # not have separate cflags (or the concept of modular build targets) with the
  # Makefile, -Wno-shadow will be used for everything.

  PLATFORM_FLAGS = \
    -DNNLIB_HIFI5 \
    -Wno-shadow

  CCFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += $(PLATFORM_FLAGS)

  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/xa_nnlib_hifi5
  NDSPLIB_PATH := $(MAKEFILE_DIR)/downloads/ndsplib-hifi5

  THIRD_PARTY_KERNEL_CC_SRCS += \
    $(shell find $(NNLIB_PATH) -name "*.c")

  # The NDSP library has a lot of file. Add as needed.
  THIRD_PARTY_KERNEL_CC_SRCS += \
    $(shell find $(NDSPLIB_PATH)/library/fft/fft -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library/fft/fft_ie -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library/fft/fft_ief -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library/twiddles -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library -name "version.c")

  EXCLUDED_NNLIB_SRCS = \
    $(NNLIB_PATH)/algo/layers/cnn/src/xa_nn_cnn_api.c \
    $(NNLIB_PATH)/algo/layers/gru/src/xa_nn_gru_api.c \
    $(NNLIB_PATH)/algo/layers/lstm/src/xa_nn_lstm_api.c

  THIRD_PARTY_KERNEL_CC_SRCS := $(filter-out $(EXCLUDED_NNLIB_SRCS), $(THIRD_PARTY_KERNEL_CC_SRCS))

  THIRD_PARTY_CC_HDRS += \
    $(shell find $(NNLIB_PATH) -name "*.h") \
    $(shell find $(NDSPLIB_PATH) -name "*.h")

  INCLUDES += \
    -I$(NNLIB_PATH)/ \
    -I$(NNLIB_PATH)/algo/kernels/ \
    -I$(NNLIB_PATH)/include/nnlib/ \
    -I$(NNLIB_PATH)/include/ \
    -I$(NNLIB_PATH)/algo/common/include/ \
    -I$(NDSPLIB_PATH)/library/include/ \
    -I$(NDSPLIB_PATH)/library/include_private/
else ifeq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifi3 hifi4))
  # NNLib hifi4 also supports hifi3
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${DOWNLOADS_DIR} hifi4 $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_ndsp_download.sh ${DOWNLOADS_DIR} $(TARGET_ARCH) $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa ndsp download: $(DOWNLOAD_RESULT))
  endif

  # TODO(b/161489252): -Wno-shadow is only needed for xannlib. But since we do
  # not have separate cflags (or the concept of modular build targets) with the
  # Makefile, -Wno-shadow will be used for everything.

  PLATFORM_FLAGS = \
    -DNNLIB_V2 \
    -Wno-shadow

  CCFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += $(PLATFORM_FLAGS)

  # NNLib for hifi4 also supports hifi3
  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/xa_nnlib_hifi4
  NDSPLIB_PATH := $(MAKEFILE_DIR)/downloads/ndsplib-$(TARGET_ARCH)

  THIRD_PARTY_KERNEL_CC_SRCS += \
    $(shell find $(NNLIB_PATH) -name "*.c")

  # The NDSP library has a lot of file. Add as needed.
  THIRD_PARTY_KERNEL_CC_SRCS += \
    $(shell find $(NDSPLIB_PATH)/library/fft/fft -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library/fft/fft_ie -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library/fft/fft_ief -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library/twiddles -name "*.c") \
    $(shell find $(NDSPLIB_PATH)/library -name "version.c")

  EXCLUDED_NNLIB_SRCS = \
    $(NNLIB_PATH)/algo/layers/cnn/src/xa_nn_cnn_api.c \
    $(NNLIB_PATH)/algo/layers/gru/src/xa_nn_gru_api.c \
    $(NNLIB_PATH)/algo/layers/lstm/src/xa_nn_lstm_api.c

  ifeq ($(TARGET_ARCH), hifi3)
    EXCLUDED_NNLIB_SRCS += \
      $(NNLIB_PATH)/algo/ndsp/hifi4/src/pow2f_tbl.c \
      $(NNLIB_PATH)/algo/ndsp/hifi4/src/scl_tanhf_hifi4.c \
      $(NNLIB_PATH)/algo/ndsp/hifi4/src/vec_tanhf_hifi4.c \
      $(NNLIB_PATH)/algo/ndsp/hifi4/src/tanhf_tbl.c
  endif
  
  ifeq ($(TARGET_ARCH), hifi4)
    EXCLUDED_NNLIB_SRCS += \
      $(NNLIB_PATH)/algo/kernels/activations/hifi4/xa_nn_activations_asym8_asym8.c
  endif

  THIRD_PARTY_KERNEL_CC_SRCS := $(filter-out $(EXCLUDED_NNLIB_SRCS), $(THIRD_PARTY_KERNEL_CC_SRCS))

  THIRD_PARTY_CC_HDRS += \
    $(shell find $(NNLIB_PATH) -name "*.h") \
    $(shell find $(NDSPLIB_PATH) -name "*.h")

  INCLUDES += \
    -I$(NNLIB_PATH)/ \
    -I$(NNLIB_PATH)/algo/kernels/ \
    -I$(NNLIB_PATH)/include/nnlib/ \
    -I$(NNLIB_PATH)/include/ \
    -I$(NNLIB_PATH)/algo/common/include/ \
    -I$(NDSPLIB_PATH)/library/include/ \
    -I$(NDSPLIB_PATH)/library/include_private/

else ifeq ($(TARGET_ARCH), vision_p6)
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${DOWNLOADS_DIR} vision_p6 $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  # TODO(b/161489252): -Wno-shadow is only needed for xannlib. But since we do
  # not have separate cflags (or the concept of modular build targets) with the
  # Makefile, -Wno-shadow will be used for everything.

  PLATFORM_FLAGS = \
    -DXI_ERROR_LEVEL=XI_ERROR_LEVEL_NO_ERROR \
    -DCNNRT_PERF_LEVEL=CNNRT_PERF_LEVEL_NONE \
    -DINCLUDE_XI_CNN \
    -Wno-shadow

  CCFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += $(PLATFORM_FLAGS)

  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/xi_tflmlib_vision_p6

  THIRD_PARTY_CC_SRCS += \
    $(shell find $(NNLIB_PATH) -name "*.cc")

  INCLUDES += \
    -I$(NNLIB_PATH)/flk/include \
    -I$(NNLIB_PATH)/kernels/include/ \
    -I$(NNLIB_PATH)/runtime/include/

  LDFLAGS += -lidma
else
  $(error Unsupported TARGET_ARCH=$(TARGET_ARCH))
endif
