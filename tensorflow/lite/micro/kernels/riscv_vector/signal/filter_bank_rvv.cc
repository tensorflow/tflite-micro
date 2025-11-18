#include <riscv_vector.h>

#include "tensorflow/lite/micro/kernels/riscv_vector/signal/filter_bank_rvv.h"
#include "tensorflow/lite/micro/micro_log.h"

#define RVV_MAX_BUFFER_VL 64

void FilterbankAccumulateChannelsRVV(const FilterbankConfig* config,
                                  const uint32_t* input, uint64_t* output) {
  uint64_t unweight_accumulator = 0;

  for (int i = 0; i < config->num_channels + 1; i++) {
    const int16_t freq_start = config->channel_frequency_starts[i];
    const int16_t weight_start = config->channel_weight_starts[i];
    const int16_t channel_width = config->channel_widths[i];

    uint64_t channel_w_acc = unweight_accumulator;
    uint64_t channel_uw_acc = 0;

    if (channel_width > 0) {
      size_t vl_max_for_channel = __riscv_vsetvl_e32m4(channel_width);

      vuint32m4_t v_acc_w_low = __riscv_vmv_v_x_u32m4(0, vl_max_for_channel);
      vuint32m4_t v_acc_w_high = __riscv_vmv_v_x_u32m4(0, vl_max_for_channel);
      vuint32m4_t v_acc_uw_low = __riscv_vmv_v_x_u32m4(0, vl_max_for_channel);
      vuint32m4_t v_acc_uw_high = __riscv_vmv_v_x_u32m4(0, vl_max_for_channel);

      int j = 0;
      while (j < channel_width) {
        size_t vl = __riscv_vsetvl_e32m4(channel_width - j);

        vuint32m4_t v_input =
            __riscv_vle32_v_u32m4(&input[freq_start + j], vl);

        vuint16m2_t v_weights16 = __riscv_vle16_v_u16m2(
            reinterpret_cast<const uint16_t*>(&config->weights[weight_start + j]),
            vl);
        vuint16m2_t v_unweights16 = __riscv_vle16_v_u16m2(
            reinterpret_cast<const uint16_t*>(&config->unweights[weight_start + j]),
            vl);

        vuint32m4_t v_weights32 = __riscv_vwaddu_vx_u32m4(v_weights16, 0, vl);
        vuint32m4_t v_unweights32 =
            __riscv_vwaddu_vx_u32m4(v_unweights16, 0, vl);

        vuint32m4_t v_prod_w_low =
            __riscv_vmul_vv_u32m4(v_input, v_weights32, vl);
        vuint32m4_t v_prod_w_high =
            __riscv_vmulhu_vv_u32m4(v_input, v_weights32, vl);
        vuint32m4_t v_prod_uw_low =
            __riscv_vmul_vv_u32m4(v_input, v_unweights32, vl);
        vuint32m4_t v_prod_uw_high =
            __riscv_vmulhu_vv_u32m4(v_input, v_unweights32, vl);

        vuint32m4_t v_next_acc_w_low =
            __riscv_vadd_vv_u32m4(v_acc_w_low, v_prod_w_low, vl);
        vuint32m4_t v_next_acc_uw_low =
            __riscv_vadd_vv_u32m4(v_acc_uw_low, v_prod_uw_low, vl);

        vbool8_t v_carry_w =
            __riscv_vmsltu_vv_u32m4_b8(v_next_acc_w_low, v_acc_w_low, vl);
        vbool8_t v_carry_uw =
            __riscv_vmsltu_vv_u32m4_b8(v_next_acc_uw_low, v_acc_uw_low, vl);

        v_acc_w_high = __riscv_vadc_vvm_u32m4(v_acc_w_high, v_prod_w_high, v_carry_w, vl);
        v_acc_uw_high = __riscv_vadc_vvm_u32m4(v_acc_uw_high, v_prod_uw_high, v_carry_uw, vl);

        v_acc_w_low = v_next_acc_w_low;
        v_acc_uw_low = v_next_acc_uw_low;

        j += vl;
      }

      uint32_t acc_w_low_buf[RVV_MAX_BUFFER_VL], acc_w_high_buf[RVV_MAX_BUFFER_VL];
      uint32_t acc_uw_low_buf[RVV_MAX_BUFFER_VL], acc_uw_high_buf[RVV_MAX_BUFFER_VL];

      __riscv_vse32_v_u32m4(acc_w_low_buf, v_acc_w_low, vl_max_for_channel);
      __riscv_vse32_v_u32m4(acc_w_high_buf, v_acc_w_high, vl_max_for_channel);
      __riscv_vse32_v_u32m4(acc_uw_low_buf, v_acc_uw_low, vl_max_for_channel);
      __riscv_vse32_v_u32m4(acc_uw_high_buf, v_acc_uw_high, vl_max_for_channel);
      
      for (size_t k = 0; k < (size_t)channel_width; ++k) {
        channel_w_acc +=
            ((uint64_t)acc_w_high_buf[k] << 32) | acc_w_low_buf[k];
        channel_uw_acc +=
            ((uint64_t)acc_uw_high_buf[k] << 32) | acc_uw_low_buf[k];
      }
    }

    output[i] = channel_w_acc;
    unweight_accumulator = channel_uw_acc;
  }
}