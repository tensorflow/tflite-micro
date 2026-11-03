# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
# Regenerate with: bazel run //cmake:update_cmake_lists
# Verified by:     bazel test //cmake:cmake_lists_staleness_test

# Core unit tests

tflm_add_cc_test(
  NAME tflm_fake_micro_context_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/fake_micro_context_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_flatbuffer_utils_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/flatbuffer_utils_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_greedy_memory_planner_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/greedy_memory_planner_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_hexdump_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/hexdump_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_linear_memory_planner_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/linear_memory_planner_test.cc"
  LABELS unit core
)

if(NOT CMAKE_CROSSCOMPILING AND NOT TFLM_OPTIMIZED_KERNELS AND NOT TFLM_CO_PROCESSOR)
  tflm_add_cc_test(
    NAME tflm_memory_arena_threshold_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_arena_threshold_test.cc"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/test_conv_model.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/models/keyword_scrambled.tflite"
    LABELS unit core
  )
endif()

tflm_add_cc_test(
  NAME tflm_memory_helpers_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_helpers_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_allocation_info_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_allocation_info_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_allocator_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_allocator_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/test_conv_model.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_interpreter_context_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_interpreter_context_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_interpreter_graph_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_interpreter_graph_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_interpreter_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_interpreter_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_log_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_log_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_mutable_op_resolver_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_mutable_op_resolver_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_resource_variable_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_resource_variable_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_time_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_time_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_micro_utils_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_utils_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_non_persistent_arena_buffer_allocator_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_non_persistent_buffer_planner_shim_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_persistent_arena_buffer_allocator_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_recording_micro_allocator_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/recording_micro_allocator_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/test_conv_model.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_recording_simple_memory_allocator_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_simple_memory_allocator_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_span_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/span_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_static_vector_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/static_vector_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_test_all_op_resolver_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/test_all_op_resolver_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_testing_helpers_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing_helpers_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_util_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/util_test.cc"
  LABELS unit core
)

tflm_add_cc_test(
  NAME tflm_util_v2_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/util_v2_test.cc"
  LABELS unit core
)

# Kernel unit tests

tflm_add_cc_test(
  NAME tflm_kernel_activations_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/activations_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_add_n_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/add_n_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_add_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/add_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_arg_min_max_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/arg_min_max_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_basic_classifier_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/basic_classifier_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_batch_matmul_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/batch_matmul_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_batch_to_space_nd_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/batch_to_space_nd_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_broadcast_args_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/broadcast_args_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_broadcast_to_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/broadcast_to_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_call_once_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/call_once_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_cast_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/cast_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_ceil_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/ceil_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_circular_buffer_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/circular_buffer_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_comparisons_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/comparisons_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_concatenation_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/concatenation_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_conv_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/conv_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/testdata/conv_test_data.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_cumsum_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/cumsum_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_decode_state_huffman_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/decode_state_huffman_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_decode_state_lut_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/decode_state_lut_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_decode_state_prune_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/decode_state_prune_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_decode_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/decode_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_depth_to_space_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/depth_to_space_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_depthwise_conv_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/depthwise_conv_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_dequantize_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/dequantize_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_detection_postprocess_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/detection_postprocess_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_div_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/div_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_dynamic_update_slice_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/dynamic_update_slice_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_elementwise_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/elementwise_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_elu_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/elu_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_embedding_lookup_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/embedding_lookup_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_ethos_u_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/ethos_u_test.cc"
  ASSETS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/models/person_detect_vela.tflite"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_exp_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/exp_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_expand_dims_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/expand_dims_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_fill_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/fill_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_floor_div_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/floor_div_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_floor_mod_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/floor_mod_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_floor_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/floor_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_fully_connected_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/fully_connected_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_gather_nd_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/gather_nd_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_gather_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/gather_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_hard_swish_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/hard_swish_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_if_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/if_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_l2_pool_2d_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/l2_pool_2d_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_l2norm_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/l2norm_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_leaky_relu_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/leaky_relu_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_log_softmax_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/log_softmax_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_logical_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/logical_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_logistic_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/logistic_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_lstm_eval_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/lstm_eval_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/testdata/lstm_test_data.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_maximum_minimum_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/maximum_minimum_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_mirror_pad_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/mirror_pad_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_mul_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/mul_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_neg_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/neg_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_pack_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/pack_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_pad_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/pad_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_pooling_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/pooling_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_prelu_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/prelu_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_quantization_util_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/quantization_util_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_quantize_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/quantize_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_reduce_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/reduce_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_reshape_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/reshape_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_resize_bilinear_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/resize_bilinear_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_resize_nearest_neighbor_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/resize_nearest_neighbor_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_reverse_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/reverse_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_round_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/round_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_select_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/select_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_shape_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/shape_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_slice_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/slice_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_softmax_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/softmax_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_space_to_batch_nd_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/space_to_batch_nd_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_space_to_depth_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/space_to_depth_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_split_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/split_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_split_v_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/split_v_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_squared_difference_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/squared_difference_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_squeeze_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/squeeze_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_strided_slice_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/strided_slice_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_sub_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/sub_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_svdf_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/svdf_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_tanh_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/tanh_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_transpose_conv_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/transpose_conv_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_transpose_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/transpose_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_unidirectional_sequence_lstm_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test.cc"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/testdata/lstm_test_data.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_unpack_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/unpack_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_while_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/while_test.cc"
  LABELS unit kernel
)

tflm_add_cc_test(
  NAME tflm_kernel_zeros_like_test
  SRCS
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/zeros_like_test.cc"
  LABELS unit kernel
)

if(TFLM_ENABLE_COMPRESSION)
  tflm_add_cc_test(
    NAME tflm_kernel_decompress_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/decompress_test.cc"
    LABELS unit kernel
  )
endif()

# Signal kernel unit tests
if(TFLM_ENABLE_SIGNAL)
  tflm_add_cc_test(
    NAME tflm_signal_delay_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/delay_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/delay_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_energy_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/energy_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/energy_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_fft_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/fft_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/fft_flexbuffers_generated_data.cc"
      "${PROJECT_SOURCE_DIR}/signal/testdata/fft_test_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_filter_bank_log_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_log_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_log_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_filter_bank_spectral_subtraction_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_spectral_subtraction_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_filter_bank_square_root_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_square_root_test.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_filter_bank_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/filter_bank_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_framer_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/framer_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/framer_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_kiss_fft_wrappers_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/src/kiss_fft_wrappers/kiss_fft_wrappers_test.cc"
    DEPS tflm::signal tflm::kissfft
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_overlap_add_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/overlap_add_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/overlap_add_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_pcan_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/pcan_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/pcan_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_stacker_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/stacker_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/stacker_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )

  tflm_add_cc_test(
    NAME tflm_signal_window_test
    SRCS
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/window_test.cc"
      "${PROJECT_SOURCE_DIR}/signal/micro/kernels/window_flexbuffers_generated_data.cc"
    DEPS tflm::signal
    LABELS unit signal
  )
endif()

# Example tests
if(TFLM_BUILD_EXAMPLES)
  tflm_add_cc_test(
    NAME tflm_detection_responder_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/detection_responder_test.cc"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/detection_responder.cc"
    LABELS example
  )

  tflm_add_cc_test(
    NAME tflm_hello_world_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/hello_world/models/hello_world_float.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite"
    LABELS example
  )

  tflm_add_cc_test(
    NAME tflm_image_provider_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/image_provider_test.cc"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/image_provider.cc"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/model_settings.cc"
    LABELS example
  )

  tflm_add_cc_test(
    NAME tflm_person_detection_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/person_detection_test.cc"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/model_settings.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/testdata/no_person.bmp"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/models/person_detect.tflite"
    LABELS example
  )

  if(TFLM_ENABLE_SIGNAL)
    tflm_add_cc_test(
      NAME tflm_micro_speech_test
      SRCS
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc"
      ASSETS
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8.tflite"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized.tflite"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms.wav"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms.wav"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms.wav"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms.wav"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms.wav"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms.wav"
      DEPS tflm::signal
      LABELS example
    )
  endif()
endif()

# SeaNet integration tests
if(TFLM_BUILD_INTEGRATION_TESTS AND TFLM_ENABLE_SIGNAL)
  tflm_add_cc_test(
    NAME tflm_seanet_add_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add0_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add0_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add10.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add10_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add10_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add10_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add11.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add11_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add11_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add11_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add12.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add12_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add12_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add12_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add13.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add13_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add13_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add13_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add14.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add14_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add14_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add14_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add15.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add15_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add15_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add15_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add16.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add16_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add16_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add16_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add1_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add1_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add2_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add2_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add3_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add3_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add4_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add4_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add5.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add5_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add5_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add5_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add6.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add6_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add6_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add6_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add7.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add7_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add7_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add7_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add8.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add8_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add8_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add8_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add9.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add9_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add9_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/add/add9_input1_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_conv_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv0_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv10.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv10_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv10_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv11.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv11_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv11_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv12.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv12_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv12_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv13.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv13_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv13_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv14.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv14_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv14_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv15.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv15_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv15_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv16.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv16_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv16_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv17.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv17_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv17_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv18.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv18_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv18_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv19.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv19_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv19_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv1_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv20.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv20_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv20_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv21.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv21_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv21_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv2_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv3_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv4_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv5.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv5_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv5_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv6.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv6_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv6_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv7.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv7_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv7_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv8.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv8_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv8_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv9.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv9_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/conv/conv9_input0_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_leaky_relu_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu0_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu10.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu10_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu10_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu11.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu11_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu11_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu12.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu12_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu12_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu13.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu13_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu13_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu14.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu14_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu14_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu15.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu15_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu15_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu16.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu16_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu16_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu17.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu17_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu17_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu18.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu18_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu18_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu19.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu19_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu19_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu1_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu20.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu20_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu20_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu21.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu21_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu21_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu22.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu22_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu22_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu2_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu3_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu4_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu5.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu5_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu5_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu6.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu6_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu6_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu7.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu7_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu7_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu8.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu8_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu8_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu9.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu9_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu9_input0_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_pad_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad0_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad10.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad10_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad10_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad11.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad11_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad11_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad12.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad12_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad12_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad13.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad13_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad13_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad14.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad14_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad14_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad15.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad15_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad15_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad16.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad16_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad16_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad17.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad17_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad17_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad18.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad18_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad18_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad1_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad2_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad3_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad4_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad5.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad5_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad5_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad6.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad6_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad6_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad7.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad7_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad7_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad8.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad8_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad8_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad9.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad9_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/pad/pad9_input0_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_quantize_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_input0_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_golden_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_input0_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_strided_slice_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice0_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice10.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice10_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice10_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice11.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice11_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice11_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice12.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice12_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice12_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice13.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice13_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice13_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice14.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice14_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice14_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice15.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice15_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice15_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice16.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice16_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice16_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice17.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice17_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice17_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice18.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice18_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice18_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice19.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice19_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice19_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice1_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice20.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice20_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice20_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice21.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice21_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice21_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice22.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice22_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice22_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice23.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice23_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice23_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice24.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice24_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice24_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice25.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice25_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice25_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice26.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice26_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice26_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice27.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice27_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice27_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice28.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice28_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice28_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice29.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice29_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice29_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice2_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice30.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice30_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice30_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice31.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice31_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice31_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice32.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice32_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice32_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice33.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice33_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice33_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice3_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice5.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice5_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice5_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice6.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice6_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice6_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice7.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice7_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice7_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice8.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice8_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice8_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice9.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice9_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice9_input0_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_sub_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub0_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub0_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub1_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub1_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub2_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub2_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub3_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub3_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub4_input0_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/sub/sub4_input1_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )

  tflm_add_cc_test(
    NAME tflm_seanet_transpose_conv_integration_test
    SRCS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/integration_tests.cc"
      "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
    ASSETS
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_input0_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_input0_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_input0_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_input0_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_input1_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4.tflite"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_golden_int16.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_input0_int32.csv"
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_input1_int16.csv"
    DEPS tflm::signal
    LABELS integration seanet
  )
endif()
