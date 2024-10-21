# Enable u-arch specfic behaviours
ifneq (,$(filter $(TARGET_ARCH), x86_64))
  # CMSIS-NN optimizations not supported
endif

ifneq (,$(CMSIS_NN_LIBS))
  ifeq (,$(CMSIS_PATH))
    $(error CMSIS_NN_LIBS provided but not CMSIS_PATH)
  endif
  ifeq (,$(CMSIS_NN_PATH))
    $(error CMSIS_NN_LIBS provided but not CMSIS_NN_PATH)
  endif
endif

# Unless an external path is provided we force a download during the first
# phase of make.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${DOWNLOADS_DIR} ${TENSORFLOW_ROOT})
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif
CMSIS_NN_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/cmsis_nn
CMSIS_NN_PATH := $(CMSIS_NN_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_NN_PATH), $(CMSIS_NN_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_nn_download.sh ${DOWNLOADS_DIR} ${TENSORFLOW_ROOT})
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS-NN download: $(DOWNLOAD_RESULT))
  endif
endif

ifeq (,$(CMSIS_NN_LIBS))
  THIRD_PARTY_KERNEL_CC_SRCS += $(shell find $(CMSIS_NN_PATH)/Source -name "*.c")
else
  MICROLITE_LIBS += $(CMSIS_NN_LIBS)
endif
THIRD_PARTY_CC_HDRS += $(shell find $(CMSIS_NN_PATH)/Include -name "*.h")

# Note all the headers from CMSIS/Core/Include are needed to ensure that the
# project generation scripts copy over the compiler specific implementations of
# the various intrinisics.
THIRD_PARTY_CC_HDRS += \
  $(CMSIS_PATH)/LICENSE \
  $(CMSIS_NN_PATH)/LICENSE \
  $(wildcard $(CMSIS_PATH)/CMSIS/Core/Include/*.h)

# We add -I$(CMSIS_PATH) to enable the code in the TFLM repo (mostly in the
# tensorflow/lite/micro/kernels/cmsis_nn) to use include paths relative to
# the CMSIS code-base.
#
# The CMSIS code itself uses includes such as #include "arm_math.h" and so
# we add $(CMSIS_PATH)/CMSIS/Core/Include etc. to be able to build the CMSIS
# code without any modifications.
INCLUDES += \
  -I$(CMSIS_PATH) \
  -I$(CMSIS_NN_PATH) \
  -I$(CMSIS_PATH)/CMSIS/Core/Include \
  -I$(CMSIS_NN_PATH)/Include
