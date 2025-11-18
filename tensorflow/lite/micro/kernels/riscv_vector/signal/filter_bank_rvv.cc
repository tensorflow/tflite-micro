#include <riscv_vector.h>

#include "tensorflow/lite/micro/kernels/riscv_vector/signal/filter_bank_rvv.h"
#include "tensorflow/lite/micro/micro_log.h"

#define RVV_MAX_BUFFER_VL 64

void FilterbankAccumulateChannelsRVV(const FilterbankConfig* config,
                                  const uint32_t* input, uint64_t* output) {
  uint64_t weight_accumulator = 0;
  uint64_t unweight_accumulator = 0;

  for (int i = 0; i < config->num_channels + 1; i++) {
    const int16_t freq_start = config->channel_frequency_starts[i];
    const int16_t weight_start = config->channel_weight_starts[i];
    const int16_t channel_width = config->channel_widths[i];

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

      // Widen 16-bit weights to 32-bit
      vuint32m4_t v_weights32 = __riscv_vwaddu_vx_u32m4(v_weights16, 0, vl);
      vuint32m4_t v_unweights32 = __riscv_vwaddu_vx_u32m4(v_unweights16, 0, vl);

      // Perform 32x32 -> high/low 32-bit multiplication
      vuint32m4_t v_prod_w_low =
          __riscv_vmul_vv_u32m4(v_input, v_weights32, vl);
      vuint32m4_t v_prod_w_high =
          __riscv_vmulhu_vv_u32m4(v_input, v_weights32, vl);

      vuint32m4_t v_prod_uw_low =
          __riscv_vmul_vv_u32m4(v_input, v_unweights32, vl);
      vuint32m4_t v_prod_uw_high =
          __riscv_vmulhu_vv_u32m4(v_input, v_unweights32, vl);

      // Use fixed-size buffers for scalar reduction
      uint32_t prod_w_low_buf[RVV_MAX_BUFFER_VL];
      uint32_t prod_w_high_buf[RVV_MAX_BUFFER_VL];
      __riscv_vse32_v_u32m4(prod_w_low_buf, v_prod_w_low, vl);
      __riscv_vse32_v_u32m4(prod_w_high_buf, v_prod_w_high, vl);

      uint32_t prod_uw_low_buf[RVV_MAX_BUFFER_VL];
      uint32_t prod_uw_high_buf[RVV_MAX_BUFFER_VL];
      __riscv_vse32_v_u32m4(prod_uw_low_buf, v_prod_uw_low, vl);
      __riscv_vse32_v_u32m4(prod_uw_high_buf, v_prod_uw_high, vl);

      // Reconstruct 64-bit products and accumulate
      for (size_t k = 0; k < vl; k++) {
        uint64_t prod_w =
            ((uint64_t)prod_w_high_buf[k] << 32) | prod_w_low_buf[k];
        weight_accumulator += prod_w;

        uint64_t prod_uw =
            ((uint64_t)prod_uw_high_buf[k] << 32) | prod_uw_low_buf[k];
        unweight_accumulator += prod_uw;
      }

      j += vl;
    }

    output[i] = weight_accumulator;
    weight_accumulator = unweight_accumulator;
    unweight_accumulator = 0;
  }
}