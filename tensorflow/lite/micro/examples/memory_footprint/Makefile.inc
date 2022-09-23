BASELINE_MEMORY_FOOTPRINT_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/memory_footprint/baseline_memory_footprint.cc

BASELINE_MEMORY_FOOTPRINT_HDRS :=

INTERPRETER_MEMORY_FOOTPRINT_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/memory_footprint/interpreter_memory_footprint.cc

INTERPRETER_MEMORY_FOOTPRINT_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/benchmarks/micro_benchmark.h

MEMORY_FOOTPRINT_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/memory_footprint/models/simple_add_model.tflite

MEMORY_FOOTPRINT_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/memory_footprint/models/simple_add_model_data.cc

MEMORY_FOOTPRINT_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/memory_footprint/models/simple_add_model_data.h

# Builds standalone binaries for profiling memory footprint.
$(eval $(call microlite_test,baseline_memory_footprint,\
$(BASELINE_MEMORY_FOOTPRINT_SRCS),$(BASELINE_MEMORY_FOOTPRINT_HDRS)))

$(eval $(call microlite_test,interpreter_memory_footprint,\
$(INTERPRETER_MEMORY_FOOTPRINT_SRCS),$(INTERPRETER_MEMORY_FOOTPRINT_HDRS),$(MEMORY_FOOTPRINT_GENERATOR_INPUTS)))