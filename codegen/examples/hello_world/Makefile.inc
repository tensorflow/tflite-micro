CODEGEN_HELLO_WORLD_MODEL := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite

CODEGEN_HELLO_WORLD_SRCS := \
$(TENSORFLOW_ROOT)codegen/examples/hello_world/hello_world.cc

# Builds a standalone binary.
$(eval $(call codegen_model_binary,codegen_hello_world,hello_world_model,\
$(CODEGEN_HELLO_WORLD_MODEL),$(CODEGEN_HELLO_WORLD_SRCS),,))

