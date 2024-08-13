/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_transpose_conv_s8.c
 * Description:  s8 version of transpose convolution using symmetric quantization.
 *
 * $Date:        31 January 2024
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Basic s8 transpose convolution function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data)
{
    (void)bias_dims;

    if (ctx->buf == NULL || output_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t activation_min = transpose_conv_params->activation.min;
    const int32_t activation_max = transpose_conv_params->activation.max;

    const int32_t input_ch = input_dims->c;
    const int32_t input_size = input_dims->w * input_dims->h;

    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;

    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;

    const int32_t pad_x = transpose_conv_params->padding.w;
    const int32_t pad_y = transpose_conv_params->padding.h;
    const int32_t pad_x_offset = transpose_conv_params->padding_offsets.w;
    const int32_t pad_y_offset = transpose_conv_params->padding_offsets.h;

    const int32_t stride_x = transpose_conv_params->stride.w;
    const int32_t stride_y = transpose_conv_params->stride.h;
    const int32_t filter_size = filter_dims->w * filter_dims->h;

    const int32_t *output_multiplier = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    const int32_t out_offset = transpose_conv_params->output_offset;
    const int32_t input_offset = transpose_conv_params->input_offset;

    const int8_t *input_data_ptr = input_data;
    int8_t *output_data_ptr = output_data;

    int32_t *const col_data = (int32_t *)ctx->buf;
    const int32_t col_buf_size = arm_transpose_conv_s8_get_buffer_size(input_dims, filter_dims, output_dims);

    int32_t batch_cnt = input_dims->n;

    int32_t *const img_buf = output_ctx->buf;
    int32_t *img_buf_ptr = img_buf;

    while (batch_cnt)
    {
        if (bias_data == NULL)
        {
            arm_memset_s8((int8_t *)img_buf_ptr, 0, output_x * output_y * output_ch * sizeof(int32_t));
        }
        else
        {
            int32_t *img_data = img_buf;

            for (int i = 0; i < output_x * output_y; i++)
            {
                memcpy(img_data, bias_data, output_ch * sizeof(int32_t));
                img_data += output_ch;
            }
        }

        int32_t *col_data_ptr = col_data;
        const int8_t *filter_data_ptr = filter_data;

        arm_memset_s8((int8_t *)col_data_ptr, 0, col_buf_size);

        for (int i_output_ch = 0; i_output_ch < output_ch; i_output_ch++)
        {
            arm_nn_mat_mult_nt_t_s8_s32(input_data_ptr,
                                        filter_data_ptr,
                                        col_data_ptr,
                                        input_size,
                                        input_ch,
                                        filter_size,
                                        input_offset,
                                        output_ch);

            filter_data_ptr += (input_ch * filter_size);
            col_data_ptr++;
        }

        int32_t *col_buf = col_data;
        int32_t *img_data = img_buf_ptr;
        const int32_t col_y = (output_y + pad_y_offset + pad_y - kernel_y) / stride_y + 1;
        const int32_t col_x = (output_x + pad_x_offset + pad_x - kernel_x) / stride_x + 1;

        // Column to image
        for (int i_col_y = 0, i_pad_y = -pad_y; i_col_y < col_y; i_col_y++, i_pad_y += stride_y)
        {
            for (int i_col_x = 0, i_pad_x = -pad_x; i_col_x < col_x; i_col_x++, i_pad_x += stride_x)
            {
                int32_t *dst_data = img_data + (i_pad_y * output_x + i_pad_x) * output_ch;

                for (int32_t i_ker_y = i_pad_y; i_ker_y < i_pad_y + kernel_y; i_ker_y++)
                {
                    for (int32_t i_ker_x = i_pad_x; i_ker_x < i_pad_x + kernel_x; i_ker_x++)
                    {
                        if (i_ker_y >= 0 && i_ker_y < output_y && i_ker_x >= 0 && i_ker_x < output_x)
                        {
                            for (int i_output_ch = 0; i_output_ch < output_ch; i_output_ch++)
                            {
                                dst_data[i_output_ch] += col_buf[i_output_ch];
                            }
                        }
                        dst_data += output_ch;
                        col_buf += output_ch;
                    }
                    dst_data += (output_x - kernel_x) * output_ch;
                }
            }
        }
        img_data = img_buf_ptr;
        for (int i = 0; i < output_x * output_y; i++)
        {
#if defined(ARM_MATH_MVEI)
            int output_ch_idx = 0;
            int8_t *ip_out_data = output_data_ptr;
            for (int32_t i_channel_rmdr = output_ch; i_channel_rmdr > 0; i_channel_rmdr -= 4)
            {
                mve_pred16_t p = vctp32q((uint32_t)i_channel_rmdr);
                int32x4_t result = vldrwq_z_s32(&img_data[output_ch_idx], p);
                result = arm_requantize_mve_32x4(result,
                                                 vldrwq_z_s32(&output_multiplier[output_ch_idx], p),
                                                 vldrwq_z_s32(&output_shift[output_ch_idx], p));
                result = vaddq_n_s32(result, out_offset);
                result = vmaxq_s32(result, vdupq_n_s32(activation_min));
                result = vminq_s32(result, vdupq_n_s32(activation_max));
                vstrbq_p_s32(ip_out_data, result, p);
                ip_out_data += 4;
                output_ch_idx += 4;
            }
            output_data_ptr += output_ch;
#else
            int i_output_ch = 0;
            for (; i_output_ch < output_ch; i_output_ch++)
            {
                int32_t result =
                    arm_nn_requantize(img_data[i_output_ch], output_multiplier[i_output_ch], output_shift[i_output_ch]);
                result += out_offset;
                result = MAX(result, activation_min);
                result = MIN(result, activation_max);
                *output_data_ptr++ = (int8_t)result;
            }
#endif
            img_data += output_ch;
        }
        input_data_ptr += (input_size * input_ch);
        batch_cnt--;
    }
    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
