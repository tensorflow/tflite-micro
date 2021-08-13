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

ALL_TAGS += arc_mli

ifeq ($(BUILD_ARC_MLI),true)
  MLI_LIB_DIR ?= arc_mli_$(notdir $(basename $(TCF_FILE_NAME)))

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  MLI_LIB_DIR := $(MLI_LIB_DIR)_mli20
endif

  $(eval $(call add_third_party_download,$(EMBARC_MLI_URL),$(EMBARC_MLI_MD5),$(MLI_LIB_DIR),build_embarc_mli,$(TCF_FILE)))

  MLI_INCLUDE_FOLDER = $(MLI_LIB_DIR)/include

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  MLI_LIB = third_party/$(MLI_LIB_DIR)/bin/arc/libmli.a
  MICROLITE_LIBS += $(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/bin/arc/libmli.a
else
  MLI_LIB = third_party/$(MLI_LIB_DIR)/bin/libmli.a
  MICROLITE_LIBS += $(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/bin/libmli.a
endif

  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_LIB_DIR)/LICENSE
else
ifneq ($(ARC_MLI_PRE_COMPILED_TARGET),)
  MLI_LIB_DIR ?= arc_mli_package
  $(eval $(call add_third_party_download,$(EMBARC_MLI_PRE_COMPILED_URL),$(EMBARC_MLI_PRE_COMPILED_MD5),$(MLI_LIB_DIR),))

  MLI_INCLUDE_FOLDER = $(MLI_LIB_DIR)/include
  MLI_LIB = third_party/$(MLI_LIB_DIR)/bin/$(ARC_MLI_PRE_COMPILED_TARGET)/release/libmli.a
  MICROLITE_LIBS += $(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/bin/$(ARC_MLI_PRE_COMPILED_TARGET)/release/libmli.a

  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_LIB_DIR)/LICENSE
else 
$(error Target for pre compiled ARC MLI library is not defined)
endif
endif

  THIRD_PARTY_CC_HDRS += $(MLI_LIB)
  GENERATED_PROJECT_LIBS += $(MLI_LIB)

  INCLUDES += \
    -I$(MAKEFILE_DIR)/downloads/$(MLI_INCLUDE_FOLDER) \
    -I$(MAKEFILE_DIR)/downloads/$(MLI_INCLUDE_FOLDER)/api

  GENERATED_PROJECT_INCLUDES += \
    -I. \
    -I./third_party/$(MLI_INCLUDE_FOLDER) \
    -I./third_party/$(MLI_INCLUDE_FOLDER)/api

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_config.h \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_types.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_helpers_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_kernels_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_mov_api.h
else
  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_config.h \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_types.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_helpers_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_kernels_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_avepool_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_conv2d_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_depthwise_conv2d_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_maxpool_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_mov_api.h
endif

    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_slicers.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/arc_mli/mli_interface.h
ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_interface_mli_20.cc
else
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/arc_mli/mli_interface.cc

    MICROLITE_TEST_SRCS += $(wildcard tensorflow/lite/micro/kernels/arc_mli/*test.cc)
endif

  ARC_MLI_TESTS := conv depthwise_conv pooling fully_connected
  ARC_MLI_TESTS += $(foreach TEST,$(ARC_MLI_TESTS), $(TEST)_slicing)

generate_arc_mli_test_projects: $(foreach TEST,$(ARC_MLI_TESTS), generate_kernel_$(TEST)_test_make_project)
  
  ARC_EXTRA_APP_SETTINGS += \
    \nMLI_ONLY ?= false\n\
    \nifeq \($(DLR)\(MLI_ONLY\), true\)\
    \nCCFLAGS += -DTF_LITE_STRIP_REFERENCE_IMPL\
    \nCXXFLAGS += -DTF_LITE_STRIP_REFERENCE_IMPL\
    \nendif\n

endif # TARGET_ARCH
