# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/detection_postprocess_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.cc,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_circular_buffer_test,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.cc,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_conv_test,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_test_common.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/testdata/conv_test_data.cc,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_test.h \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/testdata/conv_test_data.h))

$(eval $(call microlite_test,kernel_transpose_conv_test,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose_conv_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_test_common.cc,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_test.h))

$(eval $(call microlite_test,kernel_lstm_eval_test,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/lstm_eval_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/testdata/lstm_test_data.cc,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/testdata/lstm_test_data.h))

$(eval $(call microlite_test,unidirectional_sequence_lstm_test,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/testdata/lstm_test_data.cc,\
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/testdata/lstm_test_data.h))

$(eval $(call microlite_test,kernel_signal_delay_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/delay_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/delay_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/delay_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_energy_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/energy_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/energy_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/energy_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_fft_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/fft_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/fft_flexbuffers_generated_data.cc \
  $(TENSORFLOW_ROOT)signal/testdata/fft_test_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/fft_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_framer_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/framer_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/framer_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/framer_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_filter_bank_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_filter_bank_log_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_log_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_log_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_log_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_filter_bank_square_root_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_square_root_test.cc))  

$(eval $(call microlite_test,kernel_signal_filter_bank_spectral_subtraction_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_overlap_add_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/overlap_add_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/overlap_add_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/overlap_add_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_pcan_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/pcan_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/pcan_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/pcan_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_stacker_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/stacker_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/stacker_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/stacker_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_window_test,\
  $(TENSORFLOW_ROOT)signal/micro/kernels/window_test.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/window_flexbuffers_generated_data.cc, \
  $(TENSORFLOW_ROOT)signal/micro/kernels/window_flexbuffers_generated_data.h))

# For kernel tests without extra dependencies (beyond libtensorflow-microlite.a),
# use simple for loop to generate their make targets in a common way.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/activations_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add_n_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/arg_min_max_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_matmul_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_to_space_nd_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/broadcast_args_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/broadcast_to_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/cast_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/ceil_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/comparisons_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/concatenation_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/cumsum_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depth_to_space_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depthwise_conv_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/dequantize_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/div_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/elementwise_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/elu_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/embedding_lookup_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/exp_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/expand_dims_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fill_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_div_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_mod_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fully_connected_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/gather_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/gather_nd_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/hard_swish_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/l2norm_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/l2_pool_2d_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/leaky_relu_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logical_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logistic_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/log_softmax_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/maximum_minimum_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mirror_pad_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mul_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/neg_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pack_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pad_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pooling_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/prelu_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/quantization_util_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/quantize_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reduce_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reshape_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/resize_bilinear_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/resize_nearest_neighbor_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/round_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/select_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/shape_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/slice_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/softmax_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/space_to_batch_nd_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/space_to_depth_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/split_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/split_v_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/squared_difference_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/squeeze_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/strided_slice_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/sub_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/svdf_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/tanh_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unpack_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/while_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/zeros_like_test.cc

ifeq ($(ENABLE_COMPRESSION), yes)
MICROLITE_KERNEL_SIMPLE_TEST_SRCS += $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/decompress_test.cc
endif

# Generate simple kernel test targets in a common way
$(foreach TEST_TARGET,$(MICROLITE_KERNEL_SIMPLE_TEST_SRCS),\
$(eval $(call microlite_test,kernel_$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))
