CODEGEN_HELLO_WORLD_SRCS := \
$(TENSORFLOW_ROOT)codegen/examples/hello_world/hello_world.cc \
$(TENSORFLOW_ROOT)codegen/examples/hello_world/hello_world_model.cc

CODEGEN_HELLO_WORLD_HDRS := \
$(TENSORFLOW_ROOT)codegen/examples/hello_world/hello_world_model.h

# Builds a standalone binary.
$(eval $(call microlite_test,codegen_hello_world,\
$(CODEGEN_HELLO_WORLD_SRCS),,))
