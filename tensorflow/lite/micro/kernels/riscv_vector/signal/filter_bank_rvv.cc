#include <riscv_vector.h>

#include "tensorflow/lite/micro/kernels/riscv_vector/signal/filter_bank_rvv.h"
#include "tensorflow/lite/micro/micro_log.h"

void FilterbankAccumulateChannelsRVV(const FilterbankConfig* config,
                                  const uint32_t* input, uint64_t* output)
{
    // Initialize unweighted accumulator for the first channel
    uint64_t unweight_accumulator = 0;

    // Loop over each channel
    for (int i = 0; i < config->num_channels + 1; i++)
    {
        // Get parameters for the current channel
        const int16_t freq_start = config->channel_frequency_starts[i];
        const int16_t weight_start = config->channel_weight_starts[i];
        const int16_t channel_width = config->channel_widths[i];

        // Initialize scalar accumulators for this channel
        uint64_t channel_w_acc = unweight_accumulator;
        uint64_t channel_uw_acc = 0;

        // Process channel only if it has non-zero width
        if (channel_width > 0)
        {
            // Set max vector length for the channel
            size_t vl_max = __riscv_vsetvl_e32m4(channel_width);

            // Initialize vector accumulators for 64-bit sums (low and high parts)
            vuint32m4_t v_acc_w_low = __riscv_vmv_v_x_u32m4(0, vl_max);
            vuint32m4_t v_acc_w_high = __riscv_vmv_v_x_u32m4(0, vl_max);
            vuint32m4_t v_acc_uw_low = __riscv_vmv_v_x_u32m4(0, vl_max);
            vuint32m4_t v_acc_uw_high = __riscv_vmv_v_x_u32m4(0, vl_max);

            // Initialize vector accumulators for carries (Optimization: avoid vcpop in loop)
            vuint32m4_t v_carry_w_acc = __riscv_vmv_v_x_u32m4(0, vl_max);
            vuint32m4_t v_carry_uw_acc = __riscv_vmv_v_x_u32m4(0, vl_max);

            // Process the channel width in vector-sized chunks (stripmining)
            int j = 0;
            while (j < channel_width)
            {
                // Set vector length for the current strip
                size_t vl = __riscv_vsetvl_e32m4(channel_width - j);

                // Load vector of input data
                vuint32m4_t v_input =
                    __riscv_vle32_v_u32m4(&input[freq_start + j], vl);

                // Load 16-bit weights and unweights
                vuint16m2_t v_weights16 = __riscv_vle16_v_u16m2(
                    reinterpret_cast<const uint16_t*>(&config->weights[weight_start + j]), vl);
                vuint16m2_t v_unweights16 = __riscv_vle16_v_u16m2(
                    reinterpret_cast<const uint16_t*>(&config->unweights[weight_start + j]), vl);

                // Widen weights and unweights to 32-bit
                vuint32m4_t v_weights32 = __riscv_vwaddu_vx_u32m4(v_weights16, 0, vl);
                vuint32m4_t v_unweights32 = __riscv_vwaddu_vx_u32m4(v_unweights16, 0, vl);

                // Perform 32x32 multiply, producing 64-bit results as low/high pairs
                vuint32m4_t v_prod_w_low = __riscv_vmul_vv_u32m4(v_input, v_weights32, vl);
                vuint32m4_t v_prod_w_high = __riscv_vmulhu_vv_u32m4(v_input, v_weights32, vl);
                vuint32m4_t v_prod_uw_low = __riscv_vmul_vv_u32m4(v_input, v_unweights32, vl);
                vuint32m4_t v_prod_uw_high = __riscv_vmulhu_vv_u32m4(v_input, v_unweights32, vl);

                // Add the low 32-bit parts of the products
                vuint32m4_t v_next_acc_w_low = __riscv_vadd_vv_u32m4(v_acc_w_low, v_prod_w_low, vl);
                vuint32m4_t v_next_acc_uw_low = __riscv_vadd_vv_u32m4(v_acc_uw_low, v_prod_uw_low, vl);

                // Detect carries from the low-part addition
                vbool8_t v_carry_w = __riscv_vmsltu_vv_u32m4_b8(v_next_acc_w_low, v_acc_w_low, vl);
                vbool8_t v_carry_uw = __riscv_vmsltu_vv_u32m4_b8(v_next_acc_uw_low, v_acc_uw_low, vl);

                // Optimization: Accumulate carries into vector register instead of scalar vcpop
                v_carry_w_acc = __riscv_vadd_vx_u32m4_m(v_carry_w, v_carry_w_acc, 1, vl);
                v_carry_uw_acc = __riscv_vadd_vx_u32m4_m(v_carry_uw, v_carry_uw_acc, 1, vl);

                // Add the high 32-bit parts of the products
                v_acc_w_high = __riscv_vadd_vv_u32m4(v_acc_w_high, v_prod_w_high, vl);
                v_acc_uw_high = __riscv_vadd_vv_u32m4(v_acc_uw_high, v_prod_uw_high, vl);

                // Update the low-part accumulators
                v_acc_w_low = v_next_acc_w_low;
                v_acc_uw_low = v_next_acc_uw_low;

                // Advance stripmining index
                j += vl;
            }

            // Initialize a zero vector for reduction
            vuint32m1_t v_zero = __riscv_vmv_v_x_u32m1(0, vl_max);

            // Reduce the 32-bit vector accumulators to scalar sums
            vuint32m1_t v_sum_w_low = __riscv_vredsum_vs_u32m4_u32m1(v_acc_w_low, v_zero, vl_max);
            vuint32m1_t v_sum_uw_low = __riscv_vredsum_vs_u32m4_u32m1(v_acc_uw_low, v_zero, vl_max);
            vuint32m1_t v_sum_w_high = __riscv_vredsum_vs_u32m4_u32m1(v_acc_w_high, v_zero, vl_max);
            vuint32m1_t v_sum_uw_high = __riscv_vredsum_vs_u32m4_u32m1(v_acc_uw_high, v_zero, vl_max);

            // Reduce the carry accumulators
            vuint32m1_t v_sum_carry_w = __riscv_vredsum_vs_u32m4_u32m1(v_carry_w_acc, v_zero, vl_max);
            vuint32m1_t v_sum_carry_uw = __riscv_vredsum_vs_u32m4_u32m1(v_carry_uw_acc, v_zero, vl_max);

            // Extract scalar results
            uint32_t final_w_low = __riscv_vmv_x_s_u32m1_u32(v_sum_w_low);
            uint32_t final_uw_low = __riscv_vmv_x_s_u32m1_u32(v_sum_uw_low);
            uint32_t final_w_high = __riscv_vmv_x_s_u32m1_u32(v_sum_w_high);
            uint32_t final_uw_high = __riscv_vmv_x_s_u32m1_u32(v_sum_uw_high);
            uint32_t w_carry_count = __riscv_vmv_x_s_u32m1_u32(v_sum_carry_w);
            uint32_t uw_carry_count = __riscv_vmv_x_s_u32m1_u32(v_sum_carry_uw);

            // Reconstruct the final 64-bit sum
            uint64_t final_w = ((uint64_t)(final_w_high + w_carry_count) << 32) | final_w_low;
            uint64_t final_uw = ((uint64_t)(final_uw_high + uw_carry_count) << 32) | final_uw_low;

            // Add the vector reduction result to the channel's scalar accumulator
            channel_w_acc += final_w;
            channel_uw_acc += final_uw;
        }

        // Store the final weighted result for this channel
        output[i] = channel_w_acc;

        // The unweighted sum from this channel becomes the starting accumulator for the next
        unweight_accumulator = channel_uw_acc;
    }
}