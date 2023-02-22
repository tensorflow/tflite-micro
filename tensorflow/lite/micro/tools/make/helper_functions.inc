DOWNLOAD_SCRIPT := $(MAKEFILE_DIR)/download_and_extract.sh

#Handles the details of calculating the size of a binary target.
#
#Arguments are:
# 1 - Name of target.
# 2 - Regular expression for symbols to remove from the size calculation.
#Calling eval on the output will create the targets that you need.
define microlite_size
size_$(1): $$($(1)_BINARY)
	$$(SIZE_SCRIPT) $$($(1)_BINARY) $2
endef

# Handles the details of generating a binary target, including specializing
# for the current platform, and generating project file targets.
#
# Note that while the function is called microlite_test, it is used for both
# test and non-test binaries.

# Files that end with _test are added as test targets (i.e. can be executed with
# make test_<target>. All others can be executed with make run_<target>
#
# Arguments are:
# 1 - Name of target.
# 2 - C/C++ source files
# 3 - C/C++ header files
# 4 - Model sources and model test inputs in.tflite, .wav, .bmp or .csv format.
# Calling eval on the output will create the targets that you need.
define microlite_test
$(1)_LOCAL_SRCS := $(2)

ifneq ($(4),)
  # Generate cc files and headers for all models and bitmaps in the test.
  GEN_RESULT := $$(shell python3 $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/generate_cc_arrays.py $$(GENERATED_SRCS_DIR) $(4))

  # The first ifneq is needed to be compatible with make versions prior to 4.2
  # which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
  # Ubuntu 18.04 only has version 4.1
  ifneq ($(.SHELLSTATUS),)
    ifneq ($$(.SHELLSTATUS),0)
      $$(error Something went wrong: $$(GEN_RESULT))
    endif
  endif

  $(1)_LOCAL_SRCS += $$(GEN_RESULT)
endif

ALL_SRCS += $$($(1)_LOCAL_SRCS)
$(1)_LOCAL_HDRS := $(3)
$(1)_LOCAL_OBJS := $$(addprefix $$(CORE_OBJDIR), \
$$(patsubst %.S,%.o,$$(patsubst %.cc,%.o,$$(patsubst %.c,%.o,$$($(1)_LOCAL_SRCS)))))
$(1)_BINARY := $$(BINDIR)$(1)
$$($(1)_BINARY): $$($(1)_LOCAL_OBJS) $$(MICROLITE_LIB_PATH)
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXXFLAGS) $$(INCLUDES) \
	-o $$($(1)_BINARY) $$($(1)_LOCAL_OBJS) \
	$$(MICROLITE_LIB_PATH) $$(LDFLAGS) $$(MICROLITE_LIBS)
$(1): $$($(1)_BINARY)
$(1)_bin: $$($(1)_BINARY).bin

MICROLITE_BUILD_TARGETS += $$($(1)_BINARY)

ifneq (,$(findstring _test,$(1)))
ifneq (,$(findstring integration_tests,$(1)))
  MICROLITE_INTEGRATION_TEST_TARGETS += test_$(1)
else ifneq (,$(findstring generated_micro_mutable_op_resolver,$(1)))
  MICROLITE_GEN_OP_RESOLVER_TEST_TARGETS += test_$(1)
else
  MICROLITE_TEST_TARGETS += test_$(1)
endif

# For bluepill, the CI build is failing due to introduction of the
# introduction of test_run_latency.sh script. Looks at
# https://b.corp.google.com/issues/268565399#comment11 for more details.
ifneq ($(TARGET), bluepill)
test_$(1):$$($(1)_BINARY)
	$(MAKEFILE_DIR)/test_latency_log.sh $(1) $$(TEST_SCRIPT) $$($(1)_BINARY) $$(TEST_PASS_STRING) $$(TARGET)
else
test_$(1):$$($(1)_BINARY)
	$$(TEST_SCRIPT) $$($(1)_BINARY) $$(TEST_PASS_STRING) $$(TARGET)
endif

else
run_$(1): $$($(1)_BINARY)
	$$(TEST_SCRIPT) $$($(1)_BINARY) non_test_binary $$(TARGET)
endif

endef

# Adds a dependency for a third-party library that needs to be downloaded from
# an external source.
# Arguments are:
# 1 - URL to download archive file from (can be .zip, .tgz, or .bz).
# 2 - MD5 sum of archive, to check integrity. Use md5sum tool to generate.
# 3 - Folder name to unpack library into, inside tf/l/x/m/t/downloads root.
# 4 - Optional patching action, must match clause in download_and_extract.sh.
# 5 - Optional patching action parameter
# These arguments are packed into a single '!' separated string, so no element
# can contain a '!'.
define add_third_party_download
THIRD_PARTY_DOWNLOADS += $(1)!$(2)!$(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/make/downloads/$(3)!$(4)!$(5)
endef

# Unpacks an entry in a list of strings created by add_third_party_download, and
# defines a dependency rule to download the library. The download_and_extract.sh
# script is used to handle to downloading and unpacking.
# 1 - Information about the library, separated by '!'s.
define create_download_rule
$(word 3, $(subst !, ,$(1))):
	$(DOWNLOAD_SCRIPT) $(subst !, ,$(1))
THIRD_PARTY_TARGETS += $(word 3, $(subst !, ,$(1)))
endef

# Recursively find all files of given pattern
# Arguments are:
# 1 - Starting path
# 2 - File pattern, e.g: *.h
recursive_find = $(wildcard $(1)$(2)) $(foreach dir,$(wildcard $(1)*),$(call recursive_find,$(dir)/,$(2)))

