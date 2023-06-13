import os

GECKO_SDK_PATH=os.environ.get("GECKO_SDK_PATH", None)
CMSIS_PATH=os.environ.get("CMSIS_PATH", None)

if GECKO_SDK_PATH is None:
    GECKO_SDK_PATH="/home/sml-app/software/gecko_sdk/"

if CMSIS_PATH is None:
    CMSIS_PATH= "/home/sml-app/tflite-micro/tensorflow/lite/micro/tools/make/downloads/cmsis/"

def get_fill_string():
    return """GECKO_SDK_PATH={GECKO_SDK_PATH}
CMSIS_PATH={CMSIS_PATH}

INCLUDES += \
-I$(CMSIS_PATH) \
-I$(CMSIS_PATH)/CMSIS/Core/Include \
-I$(CMSIS_PATH)/CMSIS/DSP/Include \
-I$(CMSIS_PATH)/CMSIS/NN/Include \
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


SILABS_CFLAGS = '-DNDEBUG=1' \
'-DTF_LITE_STATIC_MEMORY=1' \
'-DEFR32MG24B310F1536IM48=1' \
'-DSL_BOARD_NAME="BRD4001B"' \
'-DSL_BOARD_REV="A00"' \
'-DSL_COMPONENT_CATALOG_PRESENT=1' \
'-DTF_LITE_MCU_DEBUG_LOG=1' \
'-DCMSIS_NN=1' \
'-mfp16-format=ieee' 

CCFLAGS +=  $(SILABS_CFLAGS)
CXXFLAGS +=  $(SILABS_CFLAGS)

LDFLAGS += -lstdc++ \
-lgcc \
-lc \
-lnosys \
-mfpu=fpv5-sp-d16 
""".format(GECKO_SDK_PATH=GECKO_SDK_PATH, CMSIS_PATH=CMSIS_PATH)


def replace_list():
    return (["# FILL_HERE", get_fill_string()], ["-D__DSP_PRESENT=1",""], ["-D__FPU_PRESENT=1",""], ["-D__VTOR_PRESENT=1",""], ["-D__FPU_USED=1",""],["FLOAT=hard",""])


def update_makefile(tmp_file_str):
    for item, filler in replace_list():
        tmp_file_str = tmp_file_str.replace(item,filler)

    return tmp_file_str
