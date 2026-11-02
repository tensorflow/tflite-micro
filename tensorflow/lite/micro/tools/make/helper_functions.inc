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
  GEN_RESULT := $$(sort $$(addprefix $$(GENERATED_SRCS_DIR), \
    $$(patsubst %.tflite,%_model_data.cc, \
    $$(patsubst %.bmp,%_image_data.cc, \
    $$(patsubst %.wav,%_audio_data.cc, \
    $$(patsubst %.csv,%_test_data.cc, \
    $$(patsubst %.npy,%_test_data.cc,$(4))))))))
  $(1)_LOCAL_SRCS += $$(GEN_RESULT)

  ifneq ($(filter-out all microlite tflm libtflm clean clean_downloads config_info third_party_downloads list_gendir list_library_sources list_library_headers list_third_party_sources list_third_party_headers list_generator_dir,$(MAKECMDGOALS)),)
    ifneq ($(or $(if $(findstring integration_tests,$(1)),,1),$(filter integration_tests build_integration_tests build_all,$(MAKECMDGOALS)),$(findstring $(1),$(MAKECMDGOALS))),)
      # Generate cc files and headers for all models and bitmaps in the test.
      GEN_OUTPUT := $$(shell python3 $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/generate_cc_arrays.py $$(GENERATED_SRCS_DIR) $(4))

      # The first ifneq is needed to be compatible with make versions prior to 4.2
      # which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
      # Ubuntu 18.04 only has version 4.1
      ifneq ($(.SHELLSTATUS),)
        ifneq ($$(.SHELLSTATUS),0)
          $$(error Something went wrong: $$(GEN_OUTPUT))
        endif
      endif
    endif
  endif
endif

ALL_SRCS += $$($(1)_LOCAL_SRCS)
$(1)_LOCAL_HDRS := $(3)
$(1)_LOCAL_OBJS := $$(addprefix $$(CORE_OBJDIR), \
$$(patsubst %.S,%.o,$$(patsubst %.cc,%.o,$$(patsubst %.c,%.o,$$($(1)_LOCAL_SRCS)))))
ifneq ($(4),)
  $$($(1)_LOCAL_OBJS): | $$(patsubst %.cc,%.h,$$(GEN_RESULT))
endif
$(1)_BINARY := $$(BINDIR)$(1)
$$($(1)_BINARY): $$($(1)_LOCAL_OBJS) $$(MICROLITE_LIB_PATH)
	$$(Q)$(MKDIR_P) $$(dir $$@)
	$$(Q)$$(CXX) $$(CXXFLAGS) $$(INCLUDES) \
	-o $$($(1)_BINARY) $$($(1)_LOCAL_OBJS) \
	$$(MICROLITE_LIB_PATH) $$(LDFLAGS) $$(MICROLITE_LIBS)
$(1): $$($(1)_BINARY)
$(1)_bin: $$($(1)_BINARY).bin

ifneq (,$(findstring _test,$(1)))
ifneq (,$(findstring integration_tests,$(1)))
  MICROLITE_INTEGRATION_TEST_TARGETS += test_$(1)
  MICROLITE_INTEGRATION_TEST_BINARIES += $$($(1)_BINARY)
else
  MICROLITE_TEST_TARGETS += test_$(1)
  MICROLITE_TEST_BINARIES += $$($(1)_BINARY)
  MICROLITE_BUILD_TARGETS += $$($(1)_BINARY)
endif

test_$(1):$$($(1)_BINARY)
	$(MAKEFILE_DIR)/test_latency_log.sh $(1) $$(TEST_SCRIPT) $$($(1)_BINARY) $$(TEST_PASS_STRING) $$(TARGET)

else
MICROLITE_BUILD_TARGETS += $$($(1)_BINARY)
ifneq (,$(findstring benchmark,$(1)))
  MICROLITE_BENCHMARK_BINARIES += $$($(1)_BINARY)
else
  MICROLITE_EXAMPLE_BINARIES += $$($(1)_BINARY)
endif
run_$(1): $$($(1)_BINARY)
	$$(TEST_SCRIPT) $$($(1)_BINARY) non_test_binary $$(TARGET)
endif

endef

# Downloads named dependencies via download_dependencies.py during Makefile
# evaluation (unless SKIP_DOWNLOADS=1) and records standalone package names in
# TFLM_THIRD_PARTY_DEPS for the third_party_downloads target.
# Arguments:
#   1 - Space-separated dependency name(s)
#   2 - Optional extra CLI flags (e.g. --dest_name foo --cc bar)
define ensure_third_party_deps
$(if $(findstring --dest_name,$(2)),,TFLM_THIRD_PARTY_DEPS += $(1))
ifneq ($$(SKIP_DOWNLOADS), 1)
  DOWNLOAD_RESULT := $$(shell python3 $$(MAKEFILE_DIR)/download_dependencies.py \
    --downloads_dir "$$(DOWNLOADS_DIR)" \
    --tensorflow_root "$$(TENSORFLOW_ROOT)" \
    --host_os "$$(HOST_OS)" \
    --host_arch "$$(HOST_ARCH)" \
    $(2) $(1))
  ifneq ($$(DOWNLOAD_RESULT), SUCCESS)
    $$(error Something went wrong with downloading $(1): $$(DOWNLOAD_RESULT))
  endif
endif
endef

# Recursively find all files of given pattern
# Arguments are:
# 1 - Starting path
# 2 - File pattern, e.g: *.h
recursive_find = $(wildcard $(1)$(2)) $(foreach dir,$(wildcard $(1)*),$(call recursive_find,$(dir)/,$(2)))

# Generic debug target to print any variable.
# Usage: make print-VARIABLE
print-%:
	@echo '$*=$($*)'
	@echo '  origin = $(origin $*)'
	@echo '  flavor = $(flavor $*)'