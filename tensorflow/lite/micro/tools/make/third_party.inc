# Third-party configuration and download rules

# The download scripts require that the downloads directory already exist for
# improved error checking. To accomodate that, we first create a downloads
# directory.
$(shell $(MKDIR_P) $(DOWNLOADS_DIR))

# Directly download the flatbuffers library.
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/flatbuffers_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the flatbuffers download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/kissfft_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the kissfft download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/pigweed_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the pigweed download: $(DOWNLOAD_RESULT))
endif

include $(MAKEFILE_DIR)/third_party_downloads.inc
THIRD_PARTY_DOWNLOADS :=
$(eval $(call add_third_party_download,$(GEMMLOWP_URL),$(GEMMLOWP_MD5),gemmlowp,))
$(eval $(call add_third_party_download,$(RUY_URL),$(RUY_MD5),ruy,))

THIRD_PARTY_CC_SRCS :=
THIRD_PARTY_KERNEL_CC_SRCS :=

# Load custom kernels.
include $(MAKEFILE_DIR)/additional_kernels.inc
