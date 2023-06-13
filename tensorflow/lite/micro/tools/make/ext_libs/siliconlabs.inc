# Enable u-arch specfic behaviours
ifneq (,$(filter $(TARGET_ARCH), cortex_m33))
  # CMSIS-NN optimizations not supported
endif


ifeq (,$(GECKO_SDK_PATH))
$(error GECKO_SDK_PATH not defined)
endif


ifeq (,$(CMSIS_PATH))
$(error CMSIS_PATH not defined)
endif

#  SILICON LABS SPECIFIC CODE ####
INCLUDES += \
 -I$(GECKO_SDK_PATH)/platform/Device/SiliconLabs/EFR32MG24/Include \
 -I$(GECKO_SDK_PATH)/platform/common/inc \
 -I$(GECKO_SDK_PATH)/hardware/board/inc \
 -I$(GECKO_SDK_PATH)/platform/CMSIS/Include \
 -I$(GECKO_SDK_PATH)/platform/service/device_init/inc \
 -I$(GECKO_SDK_PATH)/platform/emdrv/dmadrv/inc \
 -I$(GECKO_SDK_PATH)/platform/emdrv/common/inc \
 -I$(GECKO_SDK_PATH)/platform/emlib/inc \
 -I$(GECKO_SDK_PATH)/platform/emlib/host/inc \
 -I$(GECKO_SDK_PATH)/platform/service/iostream/inc \
 -I$(GECKO_SDK_PATH)/platform/driver/mvp/inc \
 -I$(GECKO_SDK_PATH)/hardware/driver/mx25_flash_shutdown/inc/sl_mx25_flash_shutdown_eusart \
 -I$(GECKO_SDK_PATH)/platform/common/toolchain/inc \
 -I$(GECKO_SDK_PATH)/platform/service/system/inc \
 -I$(GECKO_SDK_PATH)/util/third_party/tensorflow_extra/cmsis \
 -I$(GECKO_SDK_PATH)/util/third_party/tensorflow_extra \
 -I$(GECKO_SDK_PATH)/util/third_party/tensorflow_extra/flatbuffers/include \
 -I$(GECKO_SDK_PATH)/util/third_party/tensorflow_extra/gemmlowp \
 -I$(GECKO_SDK_PATH)/util/third_party/tensorflow_extra/ruy \
 -I$(GECKO_SDK_PATH)/util/third_party/tensorflow_extra/inc \
 -I$(GECKO_SDK_PATH)/util/third_party/tflite-micro \
 -I$(GECKO_SDK_PATH)/platform/service/udelay/inc \
 -I$(GECKO_SDK_PATH)/platform/driver/mvp/config \
 -I$(GECKO_SDK_PATH)/platform/driver/mvp/inc


# We add -I$(CMSIS_PATH) to enable the code in the TFLM repo (mostly in the
# tensorflow/lite/micro/kernels/cmsis_nn) to use include paths relative to
# the CMSIS code-base.
#
# The CMSIS code itself uses includes such as #include "arm_math.h" and so
# we add $(CMSIS_PATH)/CMSIS/Core/Include etc. to be able to build the CMSIS
# code without any modifications.