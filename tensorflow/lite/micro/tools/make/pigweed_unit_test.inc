# Makefile helpers for defining tests based on Pigweed's unit test module.

PIGWEED_ROOT := tensorflow/lite/micro/tools/make/downloads/pigweed

PIGWEED_UNIT_TEST_INCLUDES := \
  -I$(PIGWEED_ROOT)/pw_unit_test/public \
  -I$(PIGWEED_ROOT)/pw_polyfill/public \
  -I$(PIGWEED_ROOT)/pw_polyfill/standard_library_public \
  -I$(PIGWEED_ROOT)/pw_polyfill/public_overrides \
  -I$(PIGWEED_ROOT)/pw_preprocessor/public \
  -I$(PIGWEED_ROOT)/pw_string/public \
  -I$(PIGWEED_ROOT)/pw_status/public \
  -I$(PIGWEED_ROOT)/pw_span/public \
  -I$(PIGWEED_ROOT)/pw_span/public_overrides

PIGWEED_UNIT_TEST_SRCS := \
  $(PIGWEED_ROOT)/pw_unit_test/framework.cc \
  $(PIGWEED_ROOT)/pw_unit_test/simple_printing_event_handler.cc \
  $(PIGWEED_ROOT)/pw_string/format.cc \
  $(PIGWEED_ROOT)/pw_string/string_builder.cc \
  $(PIGWEED_ROOT)/pw_string/type_to_string.cc \
  tensorflow/lite/micro/testing/pigweed_unit_test.cc

# Generate targets for binaries that employ Pigweed's unit test module. Call eval
# on the output to create the targets.
#
# See elsewhere the definition and uses of the underlying microlite_test for
# full details. Target basenames that end with _test are added as test targets
# (i.e., can be executed with `make test_<basename>).
#
# Arguments are:
# 1 - Basename of target
# 2 - C/C++ source files
# 3 - C/C++ header files
# 4 - Model sources and model test inputs in.tflite, .wav, .bmp or .csv format
#
define pigweed_unit_test
$(1)_BINARY := $$(BINDIR)$(1)
$$($(1)_BINARY): INCLUDES += $(PIGWEED_UNIT_TEST_INCLUDES)
$(call microlite_test,$(1),$(2) $(PIGWEED_UNIT_TEST_SRCS),$(3),$(4))
endef
