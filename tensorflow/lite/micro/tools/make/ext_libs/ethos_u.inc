# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Arm Compiler will not link the Math library (see below), therefore we're filtering it out.
# See Fatal error: L6450U: Cannot find library m:
# "Arm Compiler is designed to run in a bare metal environment,
# and automatically includes implementations of these functions,
# and so no such flag is necessary."
# https://developer.arm.com/documentation/100891/0611/troubleshooting/general-troubleshooting-advice
MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

ifneq (,$(filter $(TARGET_ARCH), x86_64))
    $(error target architecture x86_64 not supported)
endif

# Unless an external path is provided we force a download during the first phase of make so
# that the files exist prior to the call to recursive_find below. add_third_party_download
# prevents the use of wildcards and recursive_find in selecting which files to add to THIRD_PARTY_SRCS.
ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH := $(MAKEFILE_DIR)/downloads/ethos_u_core_driver
ETHOSU_DRIVER_PATH := $(ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH)
ifeq ($(ETHOSU_DRIVER_PATH), $(ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH))
  $(call $(or $(shell $(DOWNLOAD_SCRIPT) $(ETHOSU_URL) $(ETHOSU_MD5) $(ETHOSU_DRIVER_PATH) >&2 && echo SUCCESS), $(error $(DOWNLOAD_SCRIPT) failed)))
endif

THIRD_PARTY_CC_HDRS += $(call recursive_find,$(ETHOSU_DRIVER_PATH)/include,*.h)
ifeq (,$(ETHOSU_DRIVER_LIBS))
    THIRD_PARTY_CC_SRCS += $(call recursive_find,$(ETHOSU_DRIVER_PATH)/src,*.c)
else
    MICROLITE_LIBS += $(ETHOSU_DRIVER_LIBS)
endif

# Currently there is a dependency to CMSIS even without OPTIMIZED_KERNEL_DIR=cmsis_nn.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif

THIRD_PARTY_CC_HDRS += $(CMSIS_PATH)/CMSIS/Core/Include/cmsis_compiler.h

INCLUDES += -I$(ETHOSU_DRIVER_PATH)/include \
            -I$(CMSIS_PATH)/CMSIS/Core/Include

ETHOSU_FLAGS := -DETHOSU_LOG_SEVERITY=ETHOSU_LOG_WARN
ifeq ($(ETHOSU_ARCH), u55)
  ETHOSU_FLAGS += \
    -DETHOSU_ARCH=u55 \
    -DETHOSU55
else ifeq ($(ETHOSU_ARCH), u65)
  ETHOSU_FLAGS += \
    -DETHOSU_ARCH=u65 \
    -DETHOSU65
else
  $(error "ETHOSU_ARCH=$(ETHOSU_ARCH) is not supported")
endif
CCFLAGS += ${ETHOSU_FLAGS}
CXXFLAGS += ${ETHOSU_FLAGS}

# Convert downloaded person detect int8 model.
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/models/person_detect_model_data_vela.cc:
	$(info Result of person detect int8 model conversion: $(shell $(MAKEFILE_DIR)/ext_libs/person_detection_int8_vela_convert.sh ${MAKEFILE_DIR}/downloads $(GENERATED_SRCS_DIR)))
