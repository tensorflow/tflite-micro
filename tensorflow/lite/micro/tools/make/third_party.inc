# Third-party configuration and download rules

ifneq ($(filter clean_downloads,$(MAKECMDGOALS)),)
  ifneq ($(filter-out clean clean_downloads,$(MAKECMDGOALS)),)
    $(error clean_downloads cannot be combined with other targets; run clean_downloads separately)
  endif
endif

# Skip downloading dependencies if only clean targets are requested.
CLEAN_ONLY_GOALS := clean clean_downloads
ifeq ($(filter-out $(CLEAN_ONLY_GOALS),$(MAKECMDGOALS)),)
  ifneq ($(MAKECMDGOALS),)
    SKIP_DOWNLOADS := 1
  endif
endif

TFLM_THIRD_PARTY_DEPS :=
$(eval $(call ensure_third_party_deps,flatbuffers kissfft))

THIRD_PARTY_CC_SRCS :=
THIRD_PARTY_KERNEL_CC_SRCS :=

# Load custom kernels.
include $(MAKEFILE_DIR)/additional_kernels.inc
