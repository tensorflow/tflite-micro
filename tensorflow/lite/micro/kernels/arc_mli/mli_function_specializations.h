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

#include "mli_api.h"  // NOLINT

namespace tflite {

// Convolution specialized function.
typedef mli_status (*conv_func_ptr)(const mli_tensor* /*in*/,
                                    const mli_tensor* /*weights*/,
                                    const mli_tensor* /*bias*/,
                                    const mli_conv2d_cfg* /*cfg*/,
                                    mli_tensor* /*out*/);

#ifdef MLI_2_0
conv_func_ptr __attribute__((weak))
mli_krn_conv2d_hwcn(const mli_tensor* weights) {
  int filter_w = weights->shape[KRNL_W_DIM_HWCN];
  int filter_h = weights->shape[KRNL_H_DIM_HWCN];

  if (filter_w == 1 && filter_h == 1) {
    return mli_krn_conv2d_hwcn_sa8_sa8_sa32_k1x1;
  } else if (filter_w == 3 && filter_h == 3) {
    return mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3;
  } else if (filter_w == 5 && filter_h == 5) {
    return mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5;
  } else {
    return mli_krn_conv2d_hwcn_sa8_sa8_sa32;
  }
}
#else
conv_func_ptr __attribute__((weak))
mli_krn_conv2d_hwcn(const mli_tensor* weights, const mli_conv2d_cfg* cfg) {
  return mli_krn_conv2d_nhwc_sa8_sa8_sa32;
}
#endif

// Depthwise convolution specialized function.
typedef mli_status (*depthwise_func_ptr)(const mli_tensor* /*in*/,
                                         const mli_tensor* /*weights*/,
                                         const mli_tensor* /*bias*/,
                                         const mli_conv2d_cfg* /*cfg*/,
                                         mli_tensor* /*out*/);

#ifdef MLI_2_0
depthwise_func_ptr __attribute__((weak))
mli_krn_depthwise_conv2d(const mli_tensor* weights) {
  int filter_w = weights->shape[KRNL_DW_W_DIM_HW1N];
  int filter_h = weights->shape[KRNL_DW_H_DIM_HW1N];

  if (filter_w == 3 && filter_h == 3) {
    return mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k3x3;
  } else if (filter_w == 5 && filter_h == 5) {
    return mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k5x5;
  } else {
    return mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32;
  }
}
#else
depthwise_func_ptr __attribute__((weak))
mli_krn_depthwise_conv2d(const mli_tensor* weights, const mli_conv2d_cfg* cfg) {
  return mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32;
}
#endif

#ifdef MLI_2_0
depthwise_func_ptr __attribute__((weak))
mli_krn_group_conv2d(const mli_tensor* weights) {
  int filter_w = weights->shape[KRNL_DW_W_DIM_HW1N];
  int filter_h = weights->shape[KRNL_DW_H_DIM_HW1N];

  if (filter_w == 3 && filter_h == 3) {
    return mli_krn_group_conv2d_hwcn_sa8_sa8_sa32_k3x3;
  } else if (filter_w == 5 && filter_h == 5) {
    return mli_krn_group_conv2d_hwcn_sa8_sa8_sa32_k5x5;
  } else {
    return mli_krn_group_conv2d_hwcn_sa8_sa8_sa32;
  }
}
#endif

// Pooling specialized functions.
typedef mli_status (*pooling_func_ptr)(const mli_tensor* /*in*/,
                                       const mli_pool_cfg* /*cfg*/,
                                       mli_tensor* /*out*/);

#ifdef MLI_2_0
pooling_func_ptr __attribute__((weak))
mli_krn_avepool(const mli_pool_cfg* cfg) {
  int filter_w = cfg->kernel_width;
  int filter_h = cfg->kernel_height;

  if (filter_w == 2 && filter_h == 2) {
    return mli_krn_avepool_hwc_sa8_k2x2;
  } else if (filter_w == 3 && filter_h == 3) {
    return mli_krn_avepool_hwc_sa8_k3x3;
  } else {
    return mli_krn_avepool_hwc_sa8;
  }
}
#else
pooling_func_ptr __attribute__((weak))
mli_krn_avepool(const mli_pool_cfg* cfg) {
  return mli_krn_avepool_hwc_sa8;
}
#endif

#ifdef MLI_2_0
pooling_func_ptr __attribute__((weak))
mli_krn_maxpool(const mli_pool_cfg* cfg) {
  int filter_w = cfg->kernel_width;
  int filter_h = cfg->kernel_height;

  if (filter_w == 2 && filter_h == 2) {
    return mli_krn_maxpool_hwc_sa8_k2x2;
  } else if (filter_w == 3 && filter_h == 3) {
    return mli_krn_maxpool_hwc_sa8_k3x3;
  } else {
    return mli_krn_maxpool_hwc_sa8;
  }
}
#else
pooling_func_ptr __attribute__((weak))
mli_krn_maxpool(const mli_pool_cfg* cfg) {
  return mli_krn_maxpool_hwc_sa8;
}
#endif

}  // namespace tflite