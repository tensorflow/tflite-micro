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
            // Optimization: Use LMUL=2 to fit all variables in registers and avoid spilling
            size_t vl_max = __riscv_vsetvl_e32m2(channel_width);

            // Initialize vector accumulators for 64-bit sums
            vuint32m2_t v_acc_w_low = __riscv_vmv_v_x_u32m2(0, vl_max);
            vuint32m2_t v_acc_w_high = __riscv_vmv_v_x_u32m2(0, vl_max);
            vuint32m2_t v_acc_uw_low = __riscv_vmv_v_x_u32m2(0, vl_max);
            vuint32m2_t v_acc_uw_high = __riscv_vmv_v_x_u32m2(0, vl_max);

            // Process the channel width in vector-sized chunks
            int j = 0;
            while (j < channel_width)
            {
                // Set vector length for the current strip
                size_t vl = __riscv_vsetvl_e32m2(channel_width - j);

                // Load vector of input data
                vuint32m2_t v_input = __riscv_vle32_v_u32m2(&input[freq_start + j], vl);

                // Load Weights and Unweights
                vint16m1_t v_weights16 = __riscv_vle16_v_i16m1(
                    reinterpret_cast<const int16_t*>(&config->weights[weight_start + j]), vl);
                vint16m1_t v_unweights16 = __riscv_vle16_v_i16m1(
                    reinterpret_cast<const int16_t*>(&config->unweights[weight_start + j]), vl);

                // Sign-extend weights to 32-bit
                vint32m2_t v_weights32 = __riscv_vsext_vf2_i32m2(v_weights16, vl);
                vint32m2_t v_unweights32 = __riscv_vsext_vf2_i32m2(v_unweights16, vl);

                // Reinterpret weights as unsigned bits for vmul
                vuint32m2_t v_weights32_u = __riscv_vreinterpret_v_i32m2_u32m2(v_weights32);
                vuint32m2_t v_unweights32_u = __riscv_vreinterpret_v_i32m2_u32m2(v_unweights32);

                // Low part multiply
                vuint32m2_t v_prod_w_low = __riscv_vmul_vv_u32m2(v_input, v_weights32_u, vl);
                vuint32m2_t v_prod_uw_low = __riscv_vmul_vv_u32m2(v_input, v_unweights32_u, vl);

                // High part multiply
                vint32m2_t v_prod_w_high_i = __riscv_vmulhsu_vv_i32m2(v_weights32, v_input, vl);
                vint32m2_t v_prod_uw_high_i = __riscv_vmulhsu_vv_i32m2(v_unweights32, v_input, vl);
                vuint32m2_t v_prod_w_high = __riscv_vreinterpret_v_i32m2_u32m2(v_prod_w_high_i);
                vuint32m2_t v_prod_uw_high = __riscv_vreinterpret_v_i32m2_u32m2(v_prod_uw_high_i);

                // Accumulate Low part
                vuint32m2_t v_next_acc_w_low = __riscv_vadd_vv_u32m2(v_acc_w_low, v_prod_w_low, vl);
                vuint32m2_t v_next_acc_uw_low = __riscv_vadd_vv_u32m2(v_acc_uw_low, v_prod_uw_low, vl);

                // Detect Carries (if result < accumulator, we wrapped)
                vbool16_t v_carry_w = __riscv_vmsltu_vv_u32m2_b16(v_next_acc_w_low, v_acc_w_low, vl);
                vbool16_t v_carry_uw = __riscv_vmsltu_vv_u32m2_b16(v_next_acc_uw_low, v_acc_uw_low, vl);

                // Accumulate High part
                v_acc_w_high = __riscv_vadd_vv_u32m2(v_acc_w_high, v_prod_w_high, vl);
                v_acc_uw_high = __riscv_vadd_vv_u32m2(v_acc_uw_high, v_prod_uw_high, vl);

                // Apply Carry: Add 1 to high accumulator where carry is set
                v_acc_w_high = __riscv_vadd_vx_u32m2_mu(v_carry_w, v_acc_w_high, v_acc_w_high, 1, vl);
                v_acc_uw_high = __riscv_vadd_vx_u32m2_mu(v_carry_uw, v_acc_uw_high, v_acc_uw_high, 1, vl);

                // Update low accumulator
                v_acc_w_low = v_next_acc_w_low;
                v_acc_uw_low = v_next_acc_uw_low;

                // Advance stripmining index
                j += vl;
            }

            // Initialize a zero vector for reduction
            vuint32m1_t v_zero = __riscv_vmv_v_x_u32m1(0, vl_max);

            // Reduce the 32-bit vector accumulators to scalar sums
            vuint32m1_t v_sum_w_low = __riscv_vredsum_vs_u32m2_u32m1(v_acc_w_low, v_zero, vl_max);
            vuint32m1_t v_sum_w_high = __riscv_vredsum_vs_u32m2_u32m1(v_acc_w_high, v_zero, vl_max);
            vuint32m1_t v_sum_uw_low = __riscv_vredsum_vs_u32m2_u32m1(v_acc_uw_low, v_zero, vl_max);
            vuint32m1_t v_sum_uw_high = __riscv_vredsum_vs_u32m2_u32m1(v_acc_uw_high, v_zero, vl_max);

            // Extract scalar results
            uint32_t final_w_low = __riscv_vmv_x_s_u32m1_u32(v_sum_w_low);
            uint32_t final_w_high = __riscv_vmv_x_s_u32m1_u32(v_sum_w_high);
            uint32_t final_uw_low = __riscv_vmv_x_s_u32m1_u32(v_sum_uw_low);
            uint32_t final_uw_high = __riscv_vmv_x_s_u32m1_u32(v_sum_uw_high);

            // Reconstruct the final 64-bit sum and add to channel accumulator
            channel_w_acc += ((uint64_t)final_w_high << 32) | final_w_low;
            channel_uw_acc += ((uint64_t)final_uw_high << 32) | final_uw_low;
        }

        // Store the final weighted result for this channel
        output[i] = channel_w_acc;

        // The unweighted sum from this channel becomes the starting accumulator for the next
        unweight_accumulator = channel_uw_acc;
    }
}