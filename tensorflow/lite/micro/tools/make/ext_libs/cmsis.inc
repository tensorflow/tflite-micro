ifndef CMSIS_INC_
CMSIS_INC_ := 1

# Unless an external path is provided we force a download during the first
# phase of make.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
CORTEX_DFP_PATH := $(CMSIS_PATH)/Cortex_DFP

ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  $(eval $(call ensure_third_party_deps,cmsis))
endif

endif
