# Source definitions for TFLM

# The explicitly specified list of sources and headers that are shared between
# TfLite and TFLM are in the ci/sync_from_upstream_tf.sh script.
TFL_CC_SRCS := \
$(shell find $(TENSORFLOW_ROOT)tensorflow -type d \( -path $(TENSORFLOW_ROOT)tensorflow/lite/experimental -o -path $(TENSORFLOW_ROOT)tensorflow/lite/micro \) -prune -false -o -name "*.cc" -o -name "*.c")

TFL_CC_HDRS := \
$(shell find $(TENSORFLOW_ROOT)tensorflow -type d \( -path $(TENSORFLOW_ROOT)tensorflow/lite/experimental -o -path $(TENSORFLOW_ROOT)tensorflow/lite/micro \) -prune -false -o -name "*.h")

ifneq ($(BUILD_TYPE), no_tf_lite_static_memory)
  EXCLUDED_TFL_CC_SRCS := \
	$(TENSORFLOW_ROOT)tensorflow/lite/array.cc
  TFL_CC_SRCS := $(filter-out $(EXCLUDED_TFL_CC_SRCS), $(TFL_CC_SRCS))

  EXCLUDED_TFL_CC_HDRS := \
	$(TENSORFLOW_ROOT)tensorflow/lite/array.h
  TFL_CC_HDRS := $(filter-out $(EXCLUDED_TFL_CC_HDRS), $(TFL_CC_HDRS))
endif

MICROLITE_CC_BASE_SRCS := \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/*.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/*.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_planner/*.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tflite_bridge/*.cc) \
$(TFL_CC_SRCS)

MICROLITE_CC_HDRS := \
$(wildcard $(TENSORFLOW_ROOT)signal/micro/kernels/*.h) \
$(wildcard $(TENSORFLOW_ROOT)signal/src/*.h) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/*.h) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/benchmarks/*model_data.h) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/*.h) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/*.h) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_planner/*.h) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tflite_bridge/*.h) \
$(TENSORFLOW_ROOT)LICENSE \
$(TFL_CC_HDRS)

# TODO(b/165940489): Figure out how to avoid including fixed point
# platform-specific headers.
# some kiss fft source file has to be included in header sections because
# the implemenation of three different resolution fft from one single c file.
# See http://b/201319430 for additional context.
THIRD_PARTY_CC_HDRS := \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/allocator.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/array.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/base.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/buffer.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/buffer_ref.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/code_generator.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/code_generators.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/default_allocator.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/detached_buffer.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/file_manager.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flatbuffer_builder.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flatbuffers.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flex_flat_util.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flexbuffers.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/grpc.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/hash.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/idl.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/minireflect.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/reflection.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/reflection_generated.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/registry.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/stl_emulation.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/string.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/struct.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/table.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/util.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/vector.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/vector_downward.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/verifier.h \
$(DOWNLOADS_DIR)/flatbuffers/LICENSE \
$(DOWNLOADS_DIR)/gemmlowp/fixedpoint/fixedpoint.h \
$(DOWNLOADS_DIR)/gemmlowp/fixedpoint/fixedpoint_neon.h \
$(DOWNLOADS_DIR)/gemmlowp/fixedpoint/fixedpoint_sse.h \
$(DOWNLOADS_DIR)/gemmlowp/internal/detect_platform.h \
$(DOWNLOADS_DIR)/gemmlowp/LICENSE \
$(DOWNLOADS_DIR)/kissfft/COPYING \
$(DOWNLOADS_DIR)/kissfft/kiss_fft.c \
$(DOWNLOADS_DIR)/kissfft/kiss_fft.h \
$(DOWNLOADS_DIR)/kissfft/_kiss_fft_guts.h \
$(DOWNLOADS_DIR)/kissfft/tools/kiss_fftr.c \
$(DOWNLOADS_DIR)/kissfft/tools/kiss_fftr.h \
$(DOWNLOADS_DIR)/ruy/ruy/profiler/instrumentation.h

# These two must be defined before we include the target specific Makefile.inc
# because we filter out the examples that are not supported for those targets.
# See targets/xtensa_xpg_makefile.inc for an example.
#
# We limit max depth of directories to search to not include target specific
# Makefiles that are included directly by the main example Makefile. See
# examples/micro_speech/Makefile.inc for an example. At the same time, we
# search till an arbitrary depth for files named Makefile_internal.inc as a way
# to bypass this check and allow for deeper directory structures.
MICRO_LITE_EXAMPLE_TESTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/*/Makefile.inc)

# Internal examples are copied outside the TFLM repo in google.
ifneq ($(EXTERNAL_DIR),)
  MICRO_LITE_EXAMPLE_TESTS += $(call recursive_find,$(EXTERNAL_DIR)/,Makefile_internal.inc)
endif

# Kernel integration tests must be excluded on certain targets.
MICRO_LITE_INTEGRATION_TESTS += $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/*/Makefile.inc)

MICRO_LITE_GEN_MUTABLE_OP_RESOLVER_TEST += \
  $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver_test/person_detect/Makefile.inc)

OLD_MICRO_LITE_BENCHMARKS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/benchmarks/Makefile.inc)
MICRO_LITE_BENCHMARKS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/benchmarking/Makefile.inc)

MICRO_LITE_LAYER_BY_LAYER_OUTPUT := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/Makefile.inc)

MICROLITE_BENCHMARK_SRCS := \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/benchmarks/*benchmark.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/benchmarking/*benchmark.cc)

MICROLITE_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/fake_micro_context_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/flatbuffer_utils_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/hexdump_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_helpers_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_allocation_info_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_allocator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_interpreter_context_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_interpreter_graph_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_interpreter_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_log_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_mutable_op_resolver_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_resource_variable_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_time_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_utils_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/recording_micro_allocator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/span_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/static_vector_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/testing_helpers_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_planner/greedy_memory_planner_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_planner/linear_memory_planner_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim_test.cc

MICROLITE_CC_KERNEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/activations.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/activations_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add_n.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/arg_min_max.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/assign_variable.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_matmul.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_matmul_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_to_space_nd.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/broadcast_args.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/broadcast_to.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/call_once.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/cast.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/ceil.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/comparisons.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/concatenation.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/cumsum.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decode.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decode_state.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decode_state_huffman.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decode_state_lut.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decode_state_prune.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decompress.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decompress_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depth_to_space.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depthwise_conv.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/dequantize.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/dequantize_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/detection_postprocess.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/div.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/dynamic_update_slice.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/elementwise.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/elu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/embedding_lookup.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/ethosu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/exp.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/expand_dims.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fill.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_div.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_mod.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fully_connected.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fully_connected_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/gather.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/gather_nd.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/hard_swish.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/hard_swish_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/if.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/kernel_runner.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/kernel_util.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/l2norm.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/l2_pool_2d.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/leaky_relu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/leaky_relu_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logical.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logical_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logistic.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logistic_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/log_softmax.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/lstm_eval.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/lstm_eval_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/maximum_minimum.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/micro_tensor_utils.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mirror_pad.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mul.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mul_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/neg.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pack.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pad.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pad_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pooling.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pooling_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/prelu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/prelu_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/quantize.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/quantize_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/read_variable.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reduce.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reduce_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reshape.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reshape_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/resize_bilinear.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reverse.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/round.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/select.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/shape.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/slice.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/softmax.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/softmax_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/space_to_batch_nd.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/space_to_depth.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/split.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/split_v.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/squared_difference.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/squeeze.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/strided_slice.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/strided_slice_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/sub.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/sub_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/svdf.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/svdf_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/tanh.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose_conv.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unpack.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/var_handle.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/while.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/zeros_like.cc

MICROLITE_CC_SIGNAL_KERNEL_SRCS := \
$(TENSORFLOW_ROOT)signal/micro/kernels/delay.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/energy.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/fft_auto_scale_kernel.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/fft_auto_scale_common.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_log.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_square_root.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_square_root_common.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/framer.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/irfft.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/rfft.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/stacker.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/overlap_add.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/pcan.cc \
$(TENSORFLOW_ROOT)signal/micro/kernels/window.cc \
$(TENSORFLOW_ROOT)signal/src/circular_buffer.cc \
$(TENSORFLOW_ROOT)signal/src/energy.cc \
$(TENSORFLOW_ROOT)signal/src/fft_auto_scale.cc \
$(TENSORFLOW_ROOT)signal/src/filter_bank.cc \
$(TENSORFLOW_ROOT)signal/src/filter_bank_log.cc \
$(TENSORFLOW_ROOT)signal/src/filter_bank_square_root.cc \
$(TENSORFLOW_ROOT)signal/src/filter_bank_spectral_subtraction.cc \
$(TENSORFLOW_ROOT)signal/src/irfft_float.cc \
$(TENSORFLOW_ROOT)signal/src/irfft_int16.cc \
$(TENSORFLOW_ROOT)signal/src/irfft_int32.cc \
$(TENSORFLOW_ROOT)signal/src/log.cc \
$(TENSORFLOW_ROOT)signal/src/max_abs.cc \
$(TENSORFLOW_ROOT)signal/src/msb_32.cc \
$(TENSORFLOW_ROOT)signal/src/msb_64.cc \
$(TENSORFLOW_ROOT)signal/src/overlap_add.cc \
$(TENSORFLOW_ROOT)signal/src/pcan_argc_fixed.cc \
$(TENSORFLOW_ROOT)signal/src/rfft_float.cc \
$(TENSORFLOW_ROOT)signal/src/rfft_int16.cc \
$(TENSORFLOW_ROOT)signal/src/rfft_int32.cc \
$(TENSORFLOW_ROOT)signal/src/square_root_32.cc \
$(TENSORFLOW_ROOT)signal/src/square_root_64.cc \
$(TENSORFLOW_ROOT)signal/src/window.cc

MICROLITE_TEST_HDRS := \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/*.h)