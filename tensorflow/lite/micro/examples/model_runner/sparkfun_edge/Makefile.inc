ifeq ($(TARGET), sparkfun_edge)
  INCLUDES += \
    -I$(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/

  THIRD_PARTY_CC_SRCS += \
    $(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/lis2dh12_platform_apollo3.c \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_reg.c

  THIRD_PARTY_CC_HDRS += \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_platform_apollo3.h \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_reg.h
endif
