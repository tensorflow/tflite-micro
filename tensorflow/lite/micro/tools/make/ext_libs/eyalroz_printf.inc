# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/eyalroz_printf_download.sh  $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the printf download: $(DOWNLOAD_RESULT))
endif

PRINTF_PATH := $(DOWNLOADS_DIR)/eyalroz_printf
THIRD_PARTY_CC_SRCS += \
  $(PRINTF_PATH)/src/printf/printf.c
THIRD_PARTY_CC_HDRS += \
  $(PRINTF_PATH)/src/printf/printf.h

INCLUDES += \
	-I$(PRINTF_PATH)/src
