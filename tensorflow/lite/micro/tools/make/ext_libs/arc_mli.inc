# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Settings for embARC MLI library for ARC platform. 

ifeq ($(TARGET_ARCH), arc)

ifeq ($(BUILD_ARC_MLI),true)
  MLI_LIB_DIR ?= arc_mli_$(notdir $(basename $(TCF_FILE_NAME)))

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  MLI_LIB_DIR := $(MLI_LIB_DIR)_mli20
  $(eval $(call ensure_third_party_deps,embarc_mli_2_0,--dest_name "$(MLI_LIB_DIR)"))
else
  $(eval $(call ensure_third_party_deps,embarc_mli_1_1,--dest_name "$(MLI_LIB_DIR)"))
endif

  MLI_INCLUDE_FOLDER = $(MLI_LIB_DIR)/include

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  MICROLITE_LIBS += $(DOWNLOADS_DIR)/$(MLI_LIB_DIR)/bin/arc/libmli.a
else
  MICROLITE_LIBS += $(DOWNLOADS_DIR)/$(MLI_LIB_DIR)/bin/libmli.a
endif

else
ifneq ($(ARC_MLI_PRE_COMPILED_TARGET),)
  MLI_LIB_DIR ?= arc_mli_package
  ifeq ($(MLI_LIB_DIR), arc_mli_package)
    $(eval $(call ensure_third_party_deps,arc_mli_package))
  else
    $(eval $(call ensure_third_party_deps,arc_mli_package,--dest_name "$(MLI_LIB_DIR)"))
  endif

  MLI_INCLUDE_FOLDER = $(MLI_LIB_DIR)/include
  MICROLITE_LIBS += $(DOWNLOADS_DIR)/$(MLI_LIB_DIR)/bin/$(ARC_MLI_PRE_COMPILED_TARGET)/release/libmli.a

else 
$(error Target for pre compiled ARC MLI library is not defined)
endif # ARC_MLI_PRE_COMPILED_TARGET
endif # BUILD_ARC_MLI

ifeq ($(filter $(ARC_TAGS), project_generation),)

  THIRD_PARTY_CC_HDRS += \
    $(DOWNLOADS_DIR)/$(MLI_LIB_DIR)/LICENSE

  THIRD_PARTY_CC_HDRS += $(MLI_LIB)
  GENERATED_PROJECT_LIBS += $(MLI_LIB)

  INCLUDES += \
    -I$(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER) \
    -I$(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api

  GENERATED_PROJECT_INCLUDES += \
    -I. \
    -I./third_party/$(MLI_INCLUDE_FOLDER) \
    -I./third_party/$(MLI_INCLUDE_FOLDER)/api

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  THIRD_PARTY_CC_HDRS += \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/mli_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/mli_config.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/mli_types.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_helpers_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_kernels_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_mov_api.h
else
  THIRD_PARTY_CC_HDRS += \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/mli_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/mli_config.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/mli_types.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_helpers_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_kernels_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_krn_avepool_spec_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_krn_conv2d_spec_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_krn_depthwise_conv2d_spec_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_krn_maxpool_spec_api.h \
    $(DOWNLOADS_DIR)/$(MLI_INCLUDE_FOLDER)/api/mli_mov_api.h
endif

endif # project_generation

    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_slicers.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_interface.h
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_function_specializations.h
ifneq ($(filter $(ARC_TAGS), project_generation),)
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_interface_mli_20.cc
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_interface.cc
else ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_interface_mli_20.cc
else
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_interface.cc
endif

endif # TARGET_ARCH
