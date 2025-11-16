#include <riscv_vector.h>

#include "signal/src/complex.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_int16.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_common.h"

#define FIXED_POINT 16

namespace kiss_fft_fixed16 
{
#include "kiss_fft.h"
#include "tools/kiss_fftr.h"
#include "kiss_fft.c"
#include "tools/kiss_fftr.c"
}  

void kiss_fftr_rvv(kiss_fft_fixed16::kiss_fftr_cfg st, const kiss_fft_scalar* timedata,
                   kiss_fft_fixed16::kiss_fft_cpx* freqdata)
{
  // Handle inverse FFT case and perform the initial complex FFT
  if (st->substate->inverse)
  {
    return;
  }
  kiss_fft_fixed16::kiss_fft(st->substate, (const kiss_fft_fixed16::kiss_fft_cpx*)timedata, st->tmpbuf);

  // Process DC and Nyquist bins separately (scalar operations)
  const int ncfft = st->substate->nfft;
  kiss_fft_fixed16::kiss_fft_cpx tdc;
  tdc.r = st->tmpbuf[0].r;
  tdc.i = st->tmpbuf[0].i;
  C_FIXDIV(tdc, 2);
  freqdata[0].r = tdc.r + tdc.i;
  freqdata[ncfft].r = tdc.r - tdc.i;
  freqdata[0].i = 0;
  freqdata[ncfft].i = 0;

  // Initialize pointers and loop variables for the main vector processing loop
  size_t k = 1;
  const size_t loop_end = ncfft / 2;
  const int16_t* tmpbuf_base_ptr = (const int16_t*)st->tmpbuf;
  const int16_t* twiddles_base_ptr = (const int16_t*)st->super_twiddles;
  int16_t* freqdata_base_ptr = (int16_t*)freqdata;
  ptrdiff_t stride = sizeof(kiss_fft_fixed16::kiss_fft_cpx);

  // Main loop to process FFT bins in vector chunks
  while (k <= loop_end)
  {
    // Set the vector length (vl) for the current iteration
    size_t vl = __riscv_vsetvl_e16m4(loop_end - k + 1);

    // Generate index vectors for accessing fpk, fpnk, and twiddles
    vuint16m4_t v_k_indices = __riscv_vid_v_u16m4(vl);
    v_k_indices = __riscv_vadd_vx_u16m4(v_k_indices, k, vl);
    vuint16m4_t v_neg_k_indices = __riscv_vrsub_vx_u16m4(v_k_indices, ncfft, vl);
    vuint16m4_t v_twiddle_indices = __riscv_vsub_vx_u16m4(v_k_indices, 1, vl);

    // Load the 'fpk' vector using a strided load
    vint16m4_t v_fpk_r = __riscv_vlse16_v_i16m4(&tmpbuf_base_ptr[2 * k], stride, vl);
    vint16m4_t v_fpk_i = __riscv_vlse16_v_i16m4(&tmpbuf_base_ptr[2 * k + 1], stride, vl);

    // Gather the 'fpnk' vector using indexed loads
    vuint32m8_t v_tmp_r_offsets = __riscv_vwmulu_vx_u32m8(v_neg_k_indices, sizeof(kiss_fft_fixed16::kiss_fft_cpx), vl);
    vuint32m8_t v_tmp_i_offsets = __riscv_vadd_vx_u32m8(v_tmp_r_offsets, sizeof(int16_t), vl);
    vint16m4_t v_fpnk_r_raw = __riscv_vluxei32_v_i16m4(tmpbuf_base_ptr, v_tmp_r_offsets, vl);
    vint16m4_t v_fpnk_i_raw = __riscv_vluxei32_v_i16m4(tmpbuf_base_ptr, v_tmp_i_offsets, vl);

    // Gather the twiddle factors using indexed loads
    vuint32m8_t v_tw_r_offsets = __riscv_vwmulu_vx_u32m8(v_twiddle_indices, sizeof(kiss_fft_fixed16::kiss_fft_cpx), vl);
    vuint32m8_t v_tw_i_offsets = __riscv_vadd_vx_u32m8(v_tw_r_offsets, sizeof(int16_t), vl);
    vint16m4_t v_tw_r = __riscv_vluxei32_v_i16m4(twiddles_base_ptr, v_tw_r_offsets, vl);
    vint16m4_t v_tw_i = __riscv_vluxei32_v_i16m4(twiddles_base_ptr, v_tw_i_offsets, vl);

    // Perform high-precision rounding division on fpk
    const int16_t scale = 16383;
    const int32_t round_const = 16384;
    vint32m8_t v_fpk_r_32 = __riscv_vsra_vx_i32m8(
        __riscv_vadd_vx_i32m8(__riscv_vwmul_vx_i32m8(v_fpk_r, scale, vl), round_const, vl), 15, vl);
    vint32m8_t v_fpk_i_32 = __riscv_vsra_vx_i32m8(
        __riscv_vadd_vx_i32m8(__riscv_vwmul_vx_i32m8(v_fpk_i, scale, vl), round_const, vl), 15, vl);
    vint16m4_t v_fpk_r_div2 = __riscv_vnclip_wx_i16m4(v_fpk_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m4_t v_fpk_i_div2 = __riscv_vnclip_wx_i16m4(v_fpk_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Perform high-precision rounding division on fpnk (with negated imaginary part)
    vint16m4_t v_fpnk_i_neg = __riscv_vneg_v_i16m4(v_fpnk_i_raw, vl);
    vint32m8_t v_fpnk_r_32 = __riscv_vsra_vx_i32m8(
        __riscv_vadd_vx_i32m8(__riscv_vwmul_vx_i32m8(v_fpnk_r_raw, scale, vl), round_const, vl), 15, vl);
    vint32m8_t v_fpnk_i_32 = __riscv_vsra_vx_i32m8(
        __riscv_vadd_vx_i32m8(__riscv_vwmul_vx_i32m8(v_fpnk_i_neg, scale, vl), round_const, vl), 15, vl);
    vint16m4_t v_fpnk_r_div2 = __riscv_vnclip_wx_i16m4(v_fpnk_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m4_t v_fpnk_i_div2 = __riscv_vnclip_wx_i16m4(v_fpnk_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Calculate intermediate values f1k (add) and f2k (subtract)
    vint16m4_t v_f1k_r = __riscv_vadd_vv_i16m4(v_fpk_r_div2, v_fpnk_r_div2, vl);
    vint16m4_t v_f1k_i = __riscv_vadd_vv_i16m4(v_fpk_i_div2, v_fpnk_i_div2, vl);
    vint16m4_t v_f2k_r = __riscv_vsub_vv_i16m4(v_fpk_r_div2, v_fpnk_r_div2, vl);
    vint16m4_t v_f2k_i = __riscv_vsub_vv_i16m4(v_fpk_i_div2, v_fpnk_i_div2, vl);

    // Perform complex multiplication
    vint32m8_t v_ac = __riscv_vwmul_vv_i32m8(v_f2k_r, v_tw_r, vl);
    vint32m8_t v_bd = __riscv_vwmul_vv_i32m8(v_f2k_i, v_tw_i, vl);
    vint32m8_t v_ad = __riscv_vwmul_vv_i32m8(v_f2k_r, v_tw_i, vl);
    vint32m8_t v_bc = __riscv_vwmul_vv_i32m8(v_f2k_i, v_tw_r, vl);
    vint32m8_t v_tw_res_r_32 = __riscv_vssra_vx_i32m8(__riscv_vsub_vv_i32m8(v_ac, v_bd, vl), 15, __RISCV_VXRM_RNU, vl);
    vint32m8_t v_tw_res_i_32 = __riscv_vssra_vx_i32m8(__riscv_vadd_vv_i32m8(v_ad, v_bc, vl), 15, __RISCV_VXRM_RNU, vl);
    vint16m4_t v_tw_res_r = __riscv_vnclip_wx_i16m4(v_tw_res_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m4_t v_tw_res_i = __riscv_vnclip_wx_i16m4(v_tw_res_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Calculate final output vectors
    vint16m4_t v_out_k_r = __riscv_vsra_vx_i16m4(__riscv_vadd_vv_i16m4(v_f1k_r, v_tw_res_r, vl), 1, vl);
    vint16m4_t v_out_k_i = __riscv_vsra_vx_i16m4(__riscv_vadd_vv_i16m4(v_f1k_i, v_tw_res_i, vl), 1, vl);
    vint16m4_t v_out_nk_r = __riscv_vsra_vx_i16m4(__riscv_vsub_vv_i16m4(v_f1k_r, v_tw_res_r, vl), 1, vl);
    vint16m4_t v_out_nk_i = __riscv_vsra_vx_i16m4(__riscv_vsub_vv_i16m4(v_tw_res_i, v_f1k_i, vl), 1, vl);

    // Store the results using a strided store
    __riscv_vsse16_v_i16m4(&freqdata_base_ptr[2 * k], stride, v_out_k_r, vl);
    __riscv_vsse16_v_i16m4(&freqdata_base_ptr[2 * k + 1], stride, v_out_k_i, vl);

    // Scatter the results using an indexed store
    vuint32m8_t v_freq_r_offsets = __riscv_vwmulu_vx_u32m8(v_neg_k_indices, sizeof(kiss_fft_fixed16::kiss_fft_cpx), vl);
    vuint32m8_t v_freq_i_offsets = __riscv_vadd_vx_u32m8(v_freq_r_offsets, sizeof(int16_t), vl);
    __riscv_vsuxei32_v_i16m4(freqdata_base_ptr, v_freq_r_offsets, v_out_nk_r, vl);
    __riscv_vsuxei32_v_i16m4(freqdata_base_ptr, v_freq_i_offsets, v_out_nk_i, vl);

    // Advance to the next vector chunk
    k += vl;
  }
}

size_t RfftInt16GetNeededMemory(int32_t fft_length) {
  size_t state_size = 0;
  kiss_fft_fixed16::kiss_fftr_alloc(fft_length, 0, nullptr, &state_size);
  return state_size;
}

void* RfftInt16Init(int32_t fft_length, void* state, size_t state_size) {
  return kiss_fft_fixed16::kiss_fftr_alloc(fft_length, 0, state, &state_size);
}

void RfftInt16ApplyRVV(void* state, const int16_t* input,
                    Complex<int16_t>* output) {
  kiss_fftr_rvv(
      static_cast<kiss_fft_fixed16::kiss_fftr_cfg>(state),
      reinterpret_cast<const kiss_fft_scalar*>(input),
      reinterpret_cast<kiss_fft_fixed16::kiss_fft_cpx*>(output));
}