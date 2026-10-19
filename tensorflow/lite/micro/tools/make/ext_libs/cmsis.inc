ifndef CMSIS_INC_
CMSIS_INC_ := 1

# Unless an external path is provided we force a download during the first
# phase of make.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
CORTEX_DFP_PATH := $(CMSIS_PATH)/Cortex_DFP

ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif

endif
