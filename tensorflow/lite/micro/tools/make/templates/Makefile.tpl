TARGET_TOOLCHAIN_ROOT := %{TARGET_TOOLCHAIN_ROOT}%
TARGET_TOOLCHAIN_PREFIX := %{TARGET_TOOLCHAIN_PREFIX}%

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
CXX             := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)g++'
CC              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)gcc'
AS              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)as'
AR              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ar' 
LD              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ld'
NM              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)nm'
OBJDUMP         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objdump'
OBJCOPY         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objcopy'
SIZE            := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)size'

RM = rm -f
ARFLAGS := -csr
SRCS := \
%{SRCS}%

# FILL_HERE

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))

LIBRARY_OBJS := $(filter-out tensorflow/lite/micro/examples/%, $(OBJS))

CXXFLAGS += %{CXX_FLAGS}%
CCFLAGS += %{CC_FLAGS}%

LDFLAGS += %{LINKER_FLAGS}%

# library to be generated
MICROLITE_LIB = libtensorflow-microlite.a

MICROLITE_SO = libtensorflow-microlite.so

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

%{EXECUTABLE}% : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)


# Creates a tensorflow-litemicro.a which excludes any example code.
$(MICROLITE_LIB): tensorflow/lite/schema/schema_generated.h $(LIBRARY_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MICROLITE_LIB) $(LIBRARY_OBJS)

# Creates a tensorflow-litemicro.a which excludes any example code.
$(MICROLITE_SO): tensorflow/lite/schema/schema_generated.h $(LIBRARY_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -shared  -o $(MICROLITE_SO) $(LIBRARY_OBJS)

all: %{EXECUTABLE}%

lib: $(MICROLITE_LIB)

so: $(MICROLITE_SO)

clean:
	-$(RM) $(OBJS)
	-$(RM) %{EXECUTABLE}%
	-$(RM) ${MICROLITE_LIB}
