# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# This file includes kernel test targets only.

# These tests needs additional dependencies beyond libtensorflow-microlite.a.
$(eval $(call microlite_test,kernel_detection_postprocess_test,\
  tensorflow/lite/micro/kernels/detection_postprocess_test.cc \
  tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.cc,\
  tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_circular_buffer_test,\
  tensorflow/lite/micro/kernels/circular_buffer_test.cc \
  tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.cc,\
  tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_conv_test,\
  tensorflow/lite/micro/kernels/conv_test.cc \
  tensorflow/lite/micro/kernels/conv_test_common.cc \
  tensorflow/lite/micro/kernels/testdata/conv_test_data.cc,\
  tensorflow/lite/micro/kernels/conv_test.h \
  tensorflow/lite/micro/kernels/testdata/conv_test_data.h))

$(eval $(call microlite_test,kernel_transpose_conv_test,\
  tensorflow/lite/micro/kernels/transpose_conv_test.cc \
  tensorflow/lite/micro/kernels/conv_test_common.cc,\
  tensorflow/lite/micro/kernels/conv_test.h))

$(eval $(call microlite_test,unidirectional_sequence_lstm_test,\
  tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test.cc \
  tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test_config.cc,\
  tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test_config.h))

# For kernel tests without extra dependencies (beyond libtensorflow-microlite.a),
# use simple for loop to generate their make targets in a common way.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS := \
tensorflow/lite/micro/kernels/activations_test.cc \
tensorflow/lite/micro/kernels/add_test.cc \
tensorflow/lite/micro/kernels/add_n_test.cc \
tensorflow/lite/micro/kernels/arg_min_max_test.cc \
tensorflow/lite/micro/kernels/batch_to_space_nd_test.cc \
tensorflow/lite/micro/kernels/broadcast_args_test.cc \
tensorflow/lite/micro/kernels/broadcast_to_test.cc \
tensorflow/lite/micro/kernels/cast_test.cc \
tensorflow/lite/micro/kernels/ceil_test.cc \
tensorflow/lite/micro/kernels/comparisons_test.cc \
tensorflow/lite/micro/kernels/concatenation_test.cc \
tensorflow/lite/micro/kernels/cumsum_test.cc \
tensorflow/lite/micro/kernels/depth_to_space_test.cc \
tensorflow/lite/micro/kernels/depthwise_conv_test.cc \
tensorflow/lite/micro/kernels/dequantize_test.cc \
tensorflow/lite/micro/kernels/div_test.cc \
tensorflow/lite/micro/kernels/elementwise_test.cc \
tensorflow/lite/micro/kernels/elu_test.cc \
tensorflow/lite/micro/kernels/exp_test.cc \
tensorflow/lite/micro/kernels/expand_dims_test.cc \
tensorflow/lite/micro/kernels/fill_test.cc \
tensorflow/lite/micro/kernels/floor_test.cc \
tensorflow/lite/micro/kernels/floor_div_test.cc \
tensorflow/lite/micro/kernels/floor_mod_test.cc \
tensorflow/lite/micro/kernels/fully_connected_test.cc \
tensorflow/lite/micro/kernels/gather_test.cc \
tensorflow/lite/micro/kernels/gather_nd_test.cc \
tensorflow/lite/micro/kernels/hard_swish_test.cc \
tensorflow/lite/micro/kernels/l2norm_test.cc \
tensorflow/lite/micro/kernels/l2_pool_2d_test.cc \
tensorflow/lite/micro/kernels/leaky_relu_test.cc \
tensorflow/lite/micro/kernels/logical_test.cc \
tensorflow/lite/micro/kernels/logistic_test.cc \
tensorflow/lite/micro/kernels/log_softmax_test.cc \
tensorflow/lite/micro/kernels/lstm_eval_test.cc \
tensorflow/lite/micro/kernels/maximum_minimum_test.cc \
tensorflow/lite/micro/kernels/mirror_pad_test.cc \
tensorflow/lite/micro/kernels/mul_test.cc \
tensorflow/lite/micro/kernels/neg_test.cc \
tensorflow/lite/micro/kernels/pack_test.cc \
tensorflow/lite/micro/kernels/pad_test.cc \
tensorflow/lite/micro/kernels/pooling_test.cc \
tensorflow/lite/micro/kernels/prelu_test.cc \
tensorflow/lite/micro/kernels/quantization_util_test.cc \
tensorflow/lite/micro/kernels/quantize_test.cc \
tensorflow/lite/micro/kernels/reduce_test.cc \
tensorflow/lite/micro/kernels/reshape_test.cc \
tensorflow/lite/micro/kernels/resize_bilinear_test.cc \
tensorflow/lite/micro/kernels/resize_nearest_neighbor_test.cc \
tensorflow/lite/micro/kernels/round_test.cc \
tensorflow/lite/micro/kernels/shape_test.cc \
tensorflow/lite/micro/kernels/slice_test.cc \
tensorflow/lite/micro/kernels/softmax_test.cc \
tensorflow/lite/micro/kernels/space_to_batch_nd_test.cc \
tensorflow/lite/micro/kernels/space_to_depth_test.cc \
tensorflow/lite/micro/kernels/split_test.cc \
tensorflow/lite/micro/kernels/split_v_test.cc \
tensorflow/lite/micro/kernels/squared_difference_test.cc \
tensorflow/lite/micro/kernels/squeeze_test.cc \
tensorflow/lite/micro/kernels/strided_slice_test.cc \
tensorflow/lite/micro/kernels/sub_test.cc \
tensorflow/lite/micro/kernels/svdf_test.cc \
tensorflow/lite/micro/kernels/tanh_test.cc \
tensorflow/lite/micro/kernels/transpose_test.cc \
tensorflow/lite/micro/kernels/unpack_test.cc \
tensorflow/lite/micro/kernels/while_test.cc \
tensorflow/lite/micro/kernels/zeros_like_test.cc

# Generate simple kernel test targets in a common way
$(foreach TEST_TARGET,$(MICROLITE_KERNEL_SIMPLE_TEST_SRCS),\
$(eval $(call microlite_test,kernel_$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))
