NETWORK_TESTER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/network_tester/network_tester_test.cc

NETWORK_TESTER_GENERATOR_INPUTS:=

ifeq ($(CO_PROCESSOR),ethos_u)
  NETWORK_TESTER_TEST_SRCS += \
    $(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/person_detect_model_data_vela.cc

  NETWORK_TESTER_GENERATOR_INPUTS := \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/person_detection/testdata/person.bmp
endif

NETWORK_TESTER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/network_tester/network_model.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/network_tester/input_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/network_tester/expected_output_data.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/network_tester/*/Makefile.inc)

ifdef NETWORK_MODEL
  INCLUDES += -include $(NETWORK_MODEL)
endif

ifdef OUTPUT_DATA
  INCLUDES += -include $(OUTPUT_DATA)
endif

ifdef INPUT_DATA
  INCLUDES += -include $(INPUT_DATA)
endif

ifdef ARENA_SIZE
  CXXFLAGS += -DTENSOR_ARENA_SIZE=$(ARENA_SIZE)
endif

ifdef NUM_BYTES_TO_PRINT
  CXXFLAGS += -DNUM_BYTES_TO_PRINT=$(NUM_BYTES_TO_PRINT)
endif

ifeq ($(COMPARE_OUTPUT_DATA),no)
  CXXFLAGS += -DNO_COMPARE_OUTPUT_DATA
endif

ifdef NUM_INFERENCES
  CXXFLAGS += -DNUM_INFERENCES=$(NUM_INFERENCES)
endif

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,network_tester_test,\
$(NETWORK_TESTER_TEST_SRCS),$(NETWORK_TESTER_TEST_HDRS),$(NETWORK_TESTER_GENERATOR_INPUTS)))
