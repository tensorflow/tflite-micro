# Third-party configuration and download rules

# Skip downloading dependencies if only clean targets are requested.
CLEAN_ONLY_GOALS := clean clean_downloads
ifeq ($(filter-out $(CLEAN_ONLY_GOALS),$(MAKECMDGOALS)),)
  ifneq ($(MAKECMDGOALS),)
    SKIP_DOWNLOADS := 1
  endif
endif

ifneq ($(SKIP_DOWNLOADS), 1)
  DOWNLOAD_RESULT := $(shell python3 $(MAKEFILE_DIR)/download_dependencies.py --downloads_dir $(DOWNLOADS_DIR) --tensorflow_root "$(TENSORFLOW_ROOT)")
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with downloading dependencies: $(DOWNLOAD_RESULT))
  endif
endif

include $(MAKEFILE_DIR)/third_party_downloads.inc

THIRD_PARTY_DOWNLOADS :=
THIRD_PARTY_CC_SRCS :=
THIRD_PARTY_KERNEL_CC_SRCS :=

# Load custom kernels.
include $(MAKEFILE_DIR)/additional_kernels.inc
