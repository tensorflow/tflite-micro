# Third-party configuration and download rules

# The pigweed download uses git and is more complex than a simple download and
# extract. It is left as a shell script for now.
$(shell $(MKDIR_P) $(DOWNLOADS_DIR))
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/pigweed_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the pigweed download: $(DOWNLOAD_RESULT))
endif

include $(MAKEFILE_DIR)/third_party_downloads.inc

THIRD_PARTY_DOWNLOADS :=
$(eval $(call add_third_party_download,$(GEMMLOWP_URL),$(GEMMLOWP_MD5),gemmlowp,))
$(eval $(call add_third_party_download,$(RUY_URL),$(RUY_MD5),ruy,))
$(eval $(call add_third_party_download,$(FLATBUFFERS_URL),$(FLATBUFFERS_MD5),flatbuffers,patch_file,$(TENSORFLOW_ROOT)$(FLATBUFFERS_PATCH_PATH)))
$(eval $(call add_third_party_download,$(KISSFFT_URL),$(KISSFFT_MD5),kissfft,patch_file,$(TENSORFLOW_ROOT)$(KISSFFT_PATCH_PATH)))

THIRD_PARTY_CC_SRCS :=
THIRD_PARTY_KERNEL_CC_SRCS :=

# Load custom kernels.
include $(MAKEFILE_DIR)/additional_kernels.inc
