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

# 

define microlite_test

$(eval $(call generate_project_third_party_parsing))

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

$(eval $(call generate_microlite_projects,$(1),$(call specialize,$(2)),$(3)))

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

# #############################################################

# Get macros only (i.e. the ones starting with -D) from two lists and remove duplicates
getmacros = $(patsubst -D%,%,$(filter -D%,$(sort $(filter -D%, $(1)) $(filter -D%, $(2)))))

# Look for platform or target-specific implementation files to replace reference
# implementations with, given a tag. These are expected to occur in subfolders
# of a directory where a reference implementation exists, and have the same
# interface and header file. For example,
# tensorflow/lite/micro/examples/micro_speech/audio_provider.cc
# defines a module for supplying audio data, but since no platform or OS can be
# presumed, it just always returns zeroes for its samples. The MacOS-specific
# tensorflow/lite/micro/examples/micro_speech/osx/audio_provider.cc
# has an implementation that relies on CoreAudio, and there are equivalent
# versions for other operating systems.
# The specific implementation yielded by the first tag in the list that produces
# a match is returned, else the reference version if none of the tags produce a
# match.
# All lists of source files are put through this substitution process with the
# tags of their target OS and architecture, so that implementations can be added
# by simply placing them in the file tree, with no changes to the build files
# needed.
# One confusing thing about this implementation is that we're using wildcard to
# act as a 'does file exist?' function, rather than expanding an expression.
# Wildcard will return an empty string if given a plain file path with no actual
# wildcards, if the file doesn't exist, so taking the first word of the list
# between that and the reference path will pick the specialized one if it's
# available.
# Another fix is that originally if neither file existed(either the original or
# a specialized version) this would return an empty string.Because this is
# sometimes called on third party library files before they've been downloaded,
# this caused mysterious errors, so an initial if conditional was added so that
# specializations are only looked for if the original file exists.
substitute_specialized_implementation = \
  $(if $(wildcard $(1)),$(firstword $(wildcard $(dir $(1))$(2)/$(notdir $(1))) $(wildcard $(1))),$(1))
substitute_specialized_implementations = \
  $(foreach source,$(1),$(call substitute_specialized_implementation,$(source),$(2)))

# Tests and project generation targets use this entrypoint for to get the
# specialized sources. It should be avoided for any new functionality.
# The only argument is a list of file paths.
specialize = $(call substitute_specialized_implementations,$(1),$(TARGET))

# TODO(b/143904317): It would be better to have the dependency be
# THIRD_PARTY_TARGETS instead of third_party_downloads. However, that does not
# quite work for the generate_project functions.
#
# Creates a set of rules to build a standalone makefile project for an
# executable, including all of the source and header files required in a
# separate folder and a simple makefile.
# Arguments are:
# 1 - Project type (make, etc).
# 2 - Project file template name.
# 3 - Name of executable.
# 4 - List of C/C++ source files needed to build the target.
# 5 - List of C/C++ header files needed to build the target.
# 6 - Linker flags required.
# 7 - C++ compilation flags needed.
# 8 - C compilation flags needed.
# 9 - Target Toolchian root directory
# 10 - Target Toolchain prefix
# Calling eval on the output will create a <Name>_makefile target that you
# can invoke to create the standalone project.


define generate_project
$(PRJDIR)$(3)/$(1)/%: % third_party_downloads
	mkdir -p $$(dir $$@)
	cp $$< $$@

$(PRJDIR)$(3)/$(1)/third_party/%: tensorflow/lite/micro/tools/make/downloads/% third_party_downloads
	mkdir -p $$(dir $$@)	
	cp $$< $$@

$(PRJDIR)$(3)/$(1)/%: tensorflow/lite/micro/tools/make/templates/%.tpl
	@echo "Creating Make File"
	@mkdir -p $$(dir $$@)
	@sed -E 's#\%\{SRCS\}\%#$(4)#g' $$< | \
	sed -E 's#\%\{EXECUTABLE\}\%#$(3)#g' | \
	sed -E 's#\%\{LINKER_FLAGS\}\%#$(6)#g' | \
	sed -E 's#\%\{CXX_FLAGS\}\%#$(7)#g' | \
	sed -E 's#\%\{CC_FLAGS\}\%#$(8)#g' | \
	sed -E 's#\%\{TARGET_TOOLCHAIN_ROOT\}\%#$(9)#g' | \
	sed -E 's#\%\{TARGET_TOOLCHAIN_PREFIX\}\%#$(10)#g' > $$@

generate_$(3)_$(1)_project: $(addprefix $(PRJDIR)$(3)/$(1)/, $(patsubst  $(TENSORFLOW_ROOT)%, %, $(4)) $(patsubst  $(TENSORFLOW_ROOT)%, %, $(5)) $(2) Makefile)

list_$(3)_$(1)_files:
	@echo "###### List of C/C++ source files needed to build the target. ########"
	@echo $(4) 
	@echo "###### List of C/C++ header files needed to build the target. ########"
	@echo $(5)
	@echo "###### Project file template name ########"
	@echo $(2)

list_$(3)_$(1)_info:
	@echo "THIRD PARTY CC"
	@echo "$(THIRD_PARTY_CC_HDRS)"	
	@echo "INCLUDES"
	@echo "$(INCLUDES)"	
	@echo "GENERATED_PROJECT_INCLUDES"
	@echo "$(GENERATED_PROJECT_INCLUDES)"
	@echo "MICROLITE_LIBS"
	@echo $(MICROLITE_LIBS)
	@echo "###### List of C/C++ source files needed to build the target. ########"
	@echo $(4) 
	@echo "###### List of C/C++ header files needed to build the target. ########"
	@echo $(5)
	@echo "patsubst "
	echo  $(patsubst  /home/sml-app/tflite-micro/%, %, $(4))
	@echo "with prefix"
	@echo  $(addprefix $(PRJDIR)$(3)/$(1)/, $(patsubst  $(TENSORFLOW_ROOT)%, %, $(4)) $(patsubst  $(TENSORFLOW_ROOT)%, %, $(5)) $(2))
	


list_$(3)_$(1)_inputs:
	echo  $(7)

ALL_PROJECT_TARGETS += generate_$(3)_$(1)_project
endef



# Specialized version of generate_project for TF Lite Micro test targets that
# automatically includes standard library files, so you just need to pass the
# test name and any extra source files required.
# Arguments are:
# 1 - Name of test.
# 2 - C/C++ source files implementing the test.
# 3 - C/C++ header files needed for the test.
# Calling eval on the output will create targets that you can invoke to
# generate the standalone project.
define generate_microlite_projects
$(call generate_project,make,$(MAKE_PROJECT_FILES) $($(1)_MAKE_PROJECT_FILES),$(1),$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(MICROLITE_CC_KERNEL_SRCS) $(2),$(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MICROLITE_TEST_HDRS) $(3),$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS) $(GENERATED_PROJECT_INCLUDES), $(CCFLAGS) $(GENERATED_PROJECT_INCLUDES),$(TARGET_TOOLCHAIN_ROOT),$(TARGET_TOOLCHAIN_PREFIX))
endef

# Handles the details of calculating the size of a binary target.
#
# Arguments are:
# 1 - Name of target.
# 2 - Regular expression for symbols to remove from the size calculation.
# Calling eval on the output will create the targets that you need.
define microlite_size
size_$(1): $$($(1)_BINARY)
	$$(SIZE_SCRIPT) $$($(1)_BINARY) $2
endef



# Modifies the Makefile to include all third party Srcs so that generate
# projects will create a Makefile that can be immediatley compiled without
# modification
define generate_project_third_party_parsing

# Get generated src includes with update path to third party
THIRD_PARTY_CC_SRCS += $$(filter $$(RELATIVE_MAKEFILE_DIR)/downloads/%, $$(MICROLITE_CC_PROJ_SRCS))
MICROLITE_CC_SRCS := $$(filter-out $$(THIRD_PARTY_CC_SRCS), $$(MICROLITE_CC_PROJ_SRCS))
THIRD_PARTY_CC_SRCS :=  $$(sort $$(patsubst  $$(RELATIVE_MAKEFILE_DIR)/downloads/%, third_party/%,  $$(THIRD_PARTY_CC_SRCS)))

# Get generated project includes from the includes with update path to third_party
GENERATED_PROJECT_INCLUDES += $$(filter -I$$(MAKEFILE_DIR)/downloads/%, $$(INCLUDES))
GENERATED_PROJECT_INCLUDES := $$(patsubst  -I$$(MAKEFILE_DIR)/downloads/%, -Ithird_party/%,  $$(GENERATED_PROJECT_INCLUDES))
GENERATED_PROJECT_INCLUDES += $$(filter -isystem$$(MAKEFILE_DIR)/downloads/%, $$(INCLUDES))
GENERATED_PROJECT_INCLUDES := $$(sort $$(patsubst  -isystem$$(MAKEFILE_DIR)/downloads/%, -isystemthird_party/%,  $$(GENERATED_PROJECT_INCLUDES)))
GENERATED_PROJECT_INCLUDES += -I./

# We dont copy the libraries, we just want to make sure we link to them correctly.
MICROLITE_LIBS :=  $$(sort $$(patsubst  $$(RELATIVE_MAKEFILE_DIR)/downloads/%, $$(MAKEFILE_DIR)/downloads/%, $$(MICROLITE_LIBS)))
6 :=   $$(sort $$(patsubst  $$(RELATIVE_MAKEFILE_DIR)/downloads/%, $$(MAKEFILE_DIR)/downloads/%, $$(LDFLAGS)))

# Copy all third party headers that are mentioned in includes
THIRD_PARTY_CC_HDRS :=  $$(sort $$(patsubst  $$(DOWNLOADS_DIR)/%, third_party/%,  $$(THIRD_PARTY_CC_HDRS)))

# Copy all third party headers that are mentioned in includes
INCLUDE_HDRS := $$(wildcard $$(addsuffix /*.h,$$(filter $$(DOWNLOADS_DIR)/%, $$(patsubst -I%,%,$$(INCLUDES)))))
INCLUDE_HDRS += $$(wildcard $$(addsuffix /*.h,$$(filter $$(DOWNLOADS_DIR)/%,  $$(patsubst -isystem%,%,$$(INCLUDES)))))
INCLUDE_HDRS :=  $$(sort $$(INCLUDE_HDRS))
THIRD_PARTY_CC_HDRS += $ $$(sort $(patsubst  $$(DOWNLOADS_DIR)/%, third_party/%,  $$(INCLUDE_HDRS)))


endef
