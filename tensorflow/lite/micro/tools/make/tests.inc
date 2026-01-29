# Test and benchmark definitions and loading logic.

# Load the examples.
include $(MICRO_LITE_EXAMPLE_TESTS)

# Load the integration tests.
include $(MICRO_LITE_INTEGRATION_TESTS)

# Load generated micro mutable op resolver test.
include $(MICRO_LITE_GEN_MUTABLE_OP_RESOLVER_TEST)

# Load layer_by_layer_output_tool
ifneq ($(TARGET), $(filter $(TARGET), \
  bluepill cortex_m_corstone_300 riscv32_generic hexagon))
include $(MICRO_LITE_LAYER_BY_LAYER_OUTPUT)
endif

# Load the benchmarks.
include $(OLD_MICRO_LITE_BENCHMARKS)
include $(MICRO_LITE_BENCHMARKS)

# Load custom kernel tests.
include $(MAKEFILE_DIR)/additional_tests.inc

# Some tests have additional dependencies (beyond libtensorflow-microlite.a) and
# those need to be explicitly specified with their own individual call to the
# microlite_test helper function. For these tests, we also need to make sure to
# not add targets for them if they have been excluded as part of the target
# specific Makefile.
EXPLICITLY_SPECIFIED_TEST:= $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_conv_model.h
  EXPLICITLY_SPECIFIED_TEST_GENERATOR_INPUTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/models/keyword_scrambled.tflite
  $(eval $(call microlite_test,memory_arena_threshold_test, \
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS), \
  $(EXPLICITLY_SPECIFIED_TEST_GENERATOR_INPUTS)))
endif

EXPLICITLY_SPECIFIED_TEST:= $(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_allocator_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,micro_allocator_test, \
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= $(TENSORFLOW_ROOT)tensorflow/lite/micro/recording_micro_allocator_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,recording_micro_allocator_test, \
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

# For all the tests that do not have any additional dependencies, we can
# add a make target in a common way.
$(foreach TEST_TARGET,$(filter-out $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/%,$(MICROLITE_TEST_SRCS)), \
$(eval $(call microlite_test,$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))