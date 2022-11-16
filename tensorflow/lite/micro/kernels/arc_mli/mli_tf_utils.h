/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_

#include "mli_api.h"  // NOLINT
#include "mli_interface.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#define KRNL_C_DIM_NHWC 0  // output channels

namespace tflite {
namespace ops {
namespace micro {

inline void ConvertToMliTensorData(const TfLiteTensor* tfT,
                                   MliTensorInterface* mliT,
                                   bool is_bias_tensor) {
  // Data is NULL until MliTensorAttachBuffer is called.
  mliT->SetElType(tfT->type);
  if (tfT->type == kTfLiteInt8) {
    mliT->SetData<int8_t>(nullptr, tfT->bytes);
  } else if (tfT->type == kTfLiteInt32) {
    mliT->SetData<int32_t>(nullptr, tfT->bytes);
  } else {
    MicroPrintf("Wrong data type. Expected int8_t or int32_t.");
    TFLITE_ABORT;
  }
  const int32_t dims_count = GetTensorShape(tfT).DimensionsCount();
  *mliT->Rank() = is_bias_tensor ? 1 : dims_count;

  int mli_tensor_memstride = 1;
  if (is_bias_tensor) {
    mliT->Shape()[0] = GetTensorShape(tfT).Dims(dims_count - 1);
    mliT->MemStride()[0] = mli_tensor_memstride;
  } else {
    for (int i = dims_count - 1; i >= 0; --i) {
      mliT->Shape()[i] = GetTensorShape(tfT).Dims(i);
      mliT->MemStride()[i] = mli_tensor_memstride;
      mli_tensor_memstride *= GetTensorShape(tfT).Dims(i);
    }
  }
}

inline void ConvertToMliQuantParams(const TfLiteTensor* tfT,
                                    MliTensorInterface* mliT) {
  *mliT->Dim() = -1;
#ifdef MLI_2_0
  *mliT->ZeroPointCapacity() = 0;
#endif
  *mliT->ZeroPoint<int16_t*>() = tfT->params.zero_point;
  float fscale = tfT->params.scale;
  mliT->SetScale(fscale);
}

inline void ConvertToMliQuantParamsPerChannel(const TfLiteTensor* tfT,
                                              MliTensorInterface* mliT,
                                              bool is_bias_tensor) {
  // mli tensor scale and zero_point arrays should be allocated at this point
#ifdef MLI_2_0
  TFLITE_DCHECK_NE(*mliT->Scale<int16_t**>(), 0);
  TFLITE_DCHECK_NE(*mliT->ZeroPoint<int16_t**>(), 0);
#else
  TFLITE_DCHECK_NE(*mliT->Scale<int32_t**>(), 0);
  TFLITE_DCHECK_NE(*mliT->ZeroPoint<int16_t**>(), 0);
#endif

  // get per channel quantization parameters
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tfT->quantization.params);
  int32_t quantized_dimension =
      is_bias_tensor ? 0 : affine_quantization->quantized_dimension;
  const int num_channels = mliT->Shape()[quantized_dimension];

  *mliT->Dim() = quantized_dimension;

  // set capacities
#ifdef MLI_2_0
  *mliT->ScaleFracBitsCapacity() = num_channels * sizeof(int8_t);
  *mliT->ScaleCapacity() = num_channels * sizeof(int16_t);
  *mliT->ZeroPointCapacity() = num_channels * sizeof(int16_t);
#endif
  float* fscale = affine_quantization->scale->data;
  mliT->SetScalePerChannel(fscale, num_channels);

#ifdef MLI_2_0
  int16_t* zero_point = *mliT->ZeroPoint<int16_t**>();
  for (int i = 0; i < num_channels; i++) {
    zero_point[i] = tfT->params.zero_point;
  }
#endif
}

template <typename datatype>
inline void MliTensorAttachBuffer(const TfLiteEvalTensor*,
                                  const MliTensorInterface*);

template <>
inline void MliTensorAttachBuffer<int8_t>(const TfLiteEvalTensor* tfT,
                                          const MliTensorInterface* mliT) {
  // "const_cast" here used to attach const data buffer to the initially
  // non-const mli_tensor. This is required by current implementation of MLI
  // backend and planned for redesign due to this and some other aspects.
  mliT->SetData<int8_t>(
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(tfT)),
      *mliT->DataCapacity());
}

template <>
inline void MliTensorAttachBuffer<int32_t>(const TfLiteEvalTensor* tfT,
                                           const MliTensorInterface* mliT) {
  // "const_cast" here used to attach const data buffer to the initially
  // non-const mli_tensor. This is required by current implementation of MLI
  // backend and planned for redesign due to this and some other aspects.
  mliT->SetData<int32_t>(
      const_cast<int32_t*>(tflite::micro::GetTensorData<int32_t>(tfT)),
      *mliT->DataCapacity());
}

inline void ConvertToMliTensor(const TfLiteTensor* tfT,
                               MliTensorInterface* mliT) {
  ConvertToMliTensorData(tfT, mliT, false);
  ConvertToMliQuantParams(tfT, mliT);
}

inline void ConvertToMliTensorPerChannel(const TfLiteTensor* tfT,
                                         MliTensorInterface* mliT,
                                         bool is_bias_tensor) {
  ConvertToMliTensorData(tfT, mliT, is_bias_tensor);
  ConvertToMliQuantParamsPerChannel(tfT, mliT, is_bias_tensor);
}

inline void PrepareLocalTensor(mli_tensor* tensor, mli_tensor* tensor_local) {
#ifdef MLI_2_0
  int8_t* local_data = tensor_local->data.mem.pi8;
  *tensor_local = *tensor;
  tensor_local->data.mem.pi8 = local_data;
#else
  int8_t* local_data = static_cast<int8_t*>(tensor_local->data);
  *tensor_local = *tensor;
  tensor_local->data = local_data;
#endif
}

inline void AdjustBiasTensor(MliTensorInterface* bias, MliTensorInterface* in,
                             MliTensorInterface* weights) {
  int32_t quantized_dimension = *bias->Dim();
  const int num_channels =
      quantized_dimension < 0 ? 1 : bias->Shape()[quantized_dimension];
  for (int i = 0; i < num_channels; i++) {
    int32_t adjusted_bias_scale =
        (*in->Scale<int16_t*>()) * (*weights->Scale<int16_t**>())[i];
    int in_shift = *in->ScaleFracBits<int8_t*>();
    int w_shift = (*weights->ScaleFracBits<int8_t**>())[i];
    int b_shift = (*bias->ScaleFracBits<int8_t**>())[i];
    int bias_shift = in_shift + w_shift - b_shift;
    (*bias->Scale<int16_t**>())[i] =
        (int16_t)(adjusted_bias_scale >> bias_shift);
  }
}

#ifdef MLI_2_0_KRNL_TEST
// Reorder an array according to given indexes. If backward is true, order of
// index array must be reversed.
inline static void reorder(uint32_t* arr, const uint8_t index[],
                           bool backward) {
  uint32_t temp[MLI_MAX_RANK];
  for (int8_t i = 0; i < MLI_MAX_RANK; i++) {
    if (backward)
      temp[index[i]] = arr[i];
    else
      temp[i] = arr[index[i]];
  }
  for (int8_t i = 0; i < MLI_MAX_RANK; i++) {
    arr[i] = temp[i];
  }
}

// Change shape of mli tensor and recalculate mem strides.
inline void change_shape(mli_tensor* mliT, const uint8_t dim_order[]) {
  reorder(mliT->shape, dim_order, false);

  // Calculate strides for new layout
  int mli_tensor_memstride = 1;
  for (int shape_idx = mliT->rank - 1; shape_idx >= 0; --shape_idx) {
    mliT->mem_stride[shape_idx] = mli_tensor_memstride;
    mli_tensor_memstride *= mliT->shape[shape_idx];
  }
}

inline void permute_weights(const mli_tensor* weights_src,
                            const mli_permute_cfg* permute_cfg,
                            mli_tensor* weights_dst,
                            mli_data_container* buffer_data) {
  mli_tensor buffer = {};
  buffer.el_params = weights_dst->el_params;
  buffer.data = *buffer_data;
  // Compare weights tensor size and avaliable buffer capacity.
  int buffer_size = buffer_data->capacity;
  int weights_size = mli_hlp_count_elem_num(weights_src, 0) *
                     mli_hlp_tensor_element_size(weights_src);

  // Need to change shape of distanation weights buffer according to permute
  // dimensions order to calculate slice sizes
  change_shape(weights_dst, permute_cfg->perm_dim);

  if (buffer_size >= weights_size) {
    mli_mov_cfg_t copy_config;
    mli_mov_cfg_for_copy(&copy_config);
    mli_mov_tensor_sync(weights_src, &copy_config, &buffer);
    mli_krn_permute_sa8(&buffer, permute_cfg, weights_dst);
  } else {
    // Weights shape is NHWC and output (buffer) shape is HWC where N_w = C_o.
    // Buffer size (H_o * W_o) must be more or equal then the weights size (H_w
    // * W_w * C_w). So, this is the reason, why buffer size (output tensor) is
    // divided by channel shape.
    uint32_t slice_size = buffer_size / weights_src->shape[KRNL_C_DIM_NHWC];

    mli_mov_cfg_t copy_config = {};
    uint32_t src_offsets[] = {0, 0, 0, 0};
    uint32_t src_sizes[] = {0, 0, 0, 0};
    int dst_mem_stride[] = {0, 0, 0, 0};

    mli_tensor weights_dst_sub_tensor;
    mli_sub_tensor_cfg sub_tensor_cfg = {};
    sub_tensor_cfg.sub_tensor_rank = weights_src->rank;

    // Calculate dimensions for slice accroding to buffer capacity.
    // Now, after calling change_shape() function, dst weights buffer has the
    // MLI layout (HWCN). This means, the innermost dimension (N) of dst weights
    // tensor is equal to the innermost dimension of output tensor (N).
    sub_tensor_cfg.size[weights_dst->rank - 1] =
        src_sizes[weights_dst->rank - 1] = weights_src->shape[KRNL_C_DIM_NHWC];
    // Now need to calculate other shapes for weights slice. Total slice size is
    // H*W*C*N, so to calculate sizes for each axis, avaliable slice size is
    // divided by shape for each axis.
    uint32_t slice_size_left = slice_size;
    for (uint32_t i = 0; i < weights_dst->rank - 1; i++) {
      sub_tensor_cfg.size[i] = src_sizes[i] =
          slice_size_left / weights_dst->shape[i] > 0 ? weights_dst->shape[i]
                                                      : slice_size_left;
      slice_size_left /= weights_dst->shape[i];
      slice_size_left = slice_size_left > 0 ? slice_size_left : 1;
    }
    // Need to reorder src tensor sizes because it is still in TFLM format
    // (NHWC) and src_sizes array calculated as (HWCN).
    reorder(src_sizes, permute_cfg->perm_dim, true);

    sub_tensor_cfg.offset[KRNL_C_DIM_HWCN] = src_offsets[KRNL_H_DIM_HWCN] = 0;
    sub_tensor_cfg.offset[KRNL_H_DIM_HWCN] = src_offsets[KRNL_W_DIM_HWCN] = 0;
    sub_tensor_cfg.offset[KRNL_W_DIM_HWCN] = src_offsets[KRNL_D_DIM_HWCN] = 0;
    sub_tensor_cfg.offset[KRNL_D_DIM_HWCN] = src_offsets[KRNL_C_DIM_HWCN] = 0;
    do {
      do {
        do {
          do {
            mli_mov_cfg_for_slice(&copy_config, (int*)src_offsets,
                                  (int*)src_sizes, dst_mem_stride);
            mli_mov_tensor_sync(weights_src, &copy_config, &buffer);

            mli_hlp_create_subtensor(weights_dst, &sub_tensor_cfg,
                                     &weights_dst_sub_tensor);
            mli_krn_permute_sa8(&buffer, permute_cfg, &weights_dst_sub_tensor);

            // For each axis, it is necessary to recalculate the offsets and
            // slice sizes.
            sub_tensor_cfg.offset[2] = src_offsets[3] += src_sizes[3];
            src_sizes[3] =
                std::min(src_sizes[3], weights_src->shape[3] - src_offsets[3]);
          } while (src_offsets[3] < weights_src->shape[3]);

          sub_tensor_cfg.offset[1] = src_offsets[2] += src_sizes[2];
          src_sizes[2] =
              std::min(src_sizes[2], weights_src->shape[2] - src_offsets[2]);
        } while (src_offsets[2] < weights_src->shape[2]);

        sub_tensor_cfg.offset[0] = src_offsets[1] += src_sizes[1];
        src_sizes[1] =
            std::min(src_sizes[1], weights_src->shape[1] - src_offsets[1]);
      } while (src_offsets[1] < weights_src->shape[1]);

      sub_tensor_cfg.offset[3] = src_offsets[0] += src_sizes[0];
      src_sizes[0] =
          std::min(src_sizes[0], weights_src->shape[0] - src_offsets[0]);
    } while (src_offsets[0] < weights_src->shape[0]);
  }
}
#endif

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
