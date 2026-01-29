#include <riscv_vector.h>

#include "signal/src/complex.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_int16.h"
#include "signal/src/rfft.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_common.h"

#define FIXED_POINT 16

#include "kiss_fft.h"
#include "tools/kiss_fftr.h"

namespace kiss_fft_fixed16 {
#include "_kiss_fft_guts.h"
struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
#ifdef USE_SIMD
    void * pad;
#endif
};
}

static void kf_bfly2_rvv(kiss_fft_fixed16::kiss_fft_cpx* Fout,
                         const size_t fstride,
                         const kiss_fft_fixed16::kiss_fft_cfg st, size_t m)
{
  // Initialize pointers and constants
  kiss_fft_fixed16::kiss_fft_cpx* Fout2 = Fout + m;
  const int16_t* tw1_base = (const int16_t*)st->twiddles;
  int16_t* Fout_base = (int16_t*)Fout;
  int16_t* Fout2_base = (int16_t*)Fout2;
  ptrdiff_t cpx_stride = sizeof(kiss_fft_fixed16::kiss_fft_cpx);
  ptrdiff_t tw_stride = fstride * cpx_stride;
  const int16_t scale = 16383;
  const int32_t round_const = 16384;

  // Main processing loop
  size_t k = 0;
  while (k < m)
  {
    // Set the vector length for this iteration (LMUL=2)
    size_t vl = __riscv_vsetvl_e16m2(m - k);

    // Load input data vectors
    vint16m2_t v_fout_r =
        __riscv_vlse16_v_i16m2(Fout_base + 2 * k, cpx_stride, vl);
    vint16m2_t v_fout_i =
        __riscv_vlse16_v_i16m2(Fout_base + 2 * k + 1, cpx_stride, vl);
    vint16m2_t v_fout2_r =
        __riscv_vlse16_v_i16m2(Fout2_base + 2 * k, cpx_stride, vl);
    vint16m2_t v_fout2_i =
        __riscv_vlse16_v_i16m2(Fout2_base + 2 * k + 1, cpx_stride, vl);

    // Load twiddle factor vectors
    vint16m2_t v_tw_r =
        __riscv_vlse16_v_i16m2(tw1_base + (k * fstride * 2), tw_stride, vl);
    vint16m2_t v_tw_i =
        __riscv_vlse16_v_i16m2(tw1_base + (k * fstride * 2) + 1, tw_stride, vl);

    // Perform rounding division by 2 on input data
    vint32m4_t v_fout_r_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fout_r, scale, vl),
                              round_const, vl),
        15, vl);
    vint32m4_t v_fout_i_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fout_i, scale, vl),
                              round_const, vl),
        15, vl);
    vint16m2_t v_fout_r_div2 =
        __riscv_vnclip_wx_i16m2(v_fout_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_fout_i_div2 =
        __riscv_vnclip_wx_i16m2(v_fout_i_32, 0, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_fout2_r_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fout2_r, scale, vl),
                              round_const, vl),
        15, vl);
    vint32m4_t v_fout2_i_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fout2_i, scale, vl),
                              round_const, vl),
        15, vl);
    vint16m2_t v_fout2_r_div2 =
        __riscv_vnclip_wx_i16m2(v_fout2_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_fout2_i_div2 =
        __riscv_vnclip_wx_i16m2(v_fout2_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Perform complex multiplication: t = Fout2 * tw
    vint32m4_t v_ac = __riscv_vwmul_vv_i32m4(v_fout2_r_div2, v_tw_r, vl);
    vint32m4_t v_bd = __riscv_vwmul_vv_i32m4(v_fout2_i_div2, v_tw_i, vl);
    vint32m4_t v_ad = __riscv_vwmul_vv_i32m4(v_fout2_r_div2, v_tw_i, vl);
    vint32m4_t v_bc = __riscv_vwmul_vv_i32m4(v_fout2_i_div2, v_tw_r, vl);
    vint32m4_t v_t_r_32 = __riscv_vssra_vx_i32m4(
        __riscv_vsub_vv_i32m4(v_ac, v_bd, vl), 15, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_t_i_32 = __riscv_vssra_vx_i32m4(
        __riscv_vadd_vv_i32m4(v_ad, v_bc, vl), 15, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_t_r = __riscv_vnclip_wx_i16m2(v_t_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_t_i = __riscv_vnclip_wx_i16m2(v_t_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Calculate butterfly outputs: Fout = Fout + t and Fout2 = Fout - t
    vint16m2_t v_res_fout2_r = __riscv_vsub_vv_i16m2(v_fout_r_div2, v_t_r, vl);
    vint16m2_t v_res_fout2_i = __riscv_vsub_vv_i16m2(v_fout_i_div2, v_t_i, vl);
    vint16m2_t v_res_fout_r = __riscv_vadd_vv_i16m2(v_fout_r_div2, v_t_r, vl);
    vint16m2_t v_res_fout_i = __riscv_vadd_vv_i16m2(v_fout_i_div2, v_t_i, vl);

    // Store results
    __riscv_vsse16_v_i16m2(Fout_base + 2 * k, cpx_stride, v_res_fout_r, vl);
    __riscv_vsse16_v_i16m2(Fout_base + 2 * k + 1, cpx_stride, v_res_fout_i, vl);
    __riscv_vsse16_v_i16m2(Fout2_base + 2 * k, cpx_stride, v_res_fout2_r, vl);
    __riscv_vsse16_v_i16m2(Fout2_base + 2 * k + 1, cpx_stride, v_res_fout2_i, vl);

    // Advance loop counter
    k += vl;
  }
}

static void kf_bfly4_rvv(kiss_fft_fixed16::kiss_fft_cpx* Fout,
                         const size_t fstride,
                         const kiss_fft_fixed16::kiss_fft_cfg st,
                         const size_t m)
{
  // Initialize pointers and constants
  const size_t m2 = 2 * m;
  const size_t m3 = 3 * m;

  int16_t* Fout0_base = (int16_t*)(Fout);
  int16_t* Fout1_base = (int16_t*)(Fout + m);
  int16_t* Fout2_base = (int16_t*)(Fout + m2);
  int16_t* Fout3_base = (int16_t*)(Fout + m3);
  const int16_t* tw_base = (const int16_t*)st->twiddles;
  
  ptrdiff_t cpx_stride = sizeof(kiss_fft_fixed16::kiss_fft_cpx);
  ptrdiff_t tw1_stride = fstride * cpx_stride;
  ptrdiff_t tw2_stride = fstride * 2 * cpx_stride;
  ptrdiff_t tw3_stride = fstride * 3 * cpx_stride;

  const int16_t scale = 8191;
  const int32_t round_const = 16384;

  // Main processing loop
  size_t k = 0;
  while (k < m)
  {
    // Set the vector length for this iteration (LMUL=1)
    size_t vl = __riscv_vsetvl_e16m1(m - k);

    // Load input data vectors
    vint16m1_t v_f0_r =
        __riscv_vlse16_v_i16m1(Fout0_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f0_i =
        __riscv_vlse16_v_i16m1(Fout0_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f1_r =
        __riscv_vlse16_v_i16m1(Fout1_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f1_i =
        __riscv_vlse16_v_i16m1(Fout1_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f2_r =
        __riscv_vlse16_v_i16m1(Fout2_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f2_i =
        __riscv_vlse16_v_i16m1(Fout2_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f3_r =
        __riscv_vlse16_v_i16m1(Fout3_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f3_i =
        __riscv_vlse16_v_i16m1(Fout3_base + 2 * k + 1, cpx_stride, vl);

    // Perform rounding division by 4 on input data
    vint16m1_t v_f0d_r = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f0_r, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f0d_i = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f0_i, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f1d_r = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f1_r, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f1d_i = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f1_i, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f2d_r = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f2_r, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f2d_i = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f2_i, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f3d_r = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f3_r, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_f3d_i = __riscv_vnclip_wx_i16m1(
        __riscv_vsra_vx_i32m2(
            __riscv_vadd_vx_i32m2(
                __riscv_vwmul_vx_i32m2(v_f3_i, scale, vl), round_const, vl),
            15, vl),
        0, __RISCV_VXRM_RNU, vl);

    // Load twiddle factor vectors
    vint16m1_t v_tw1_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 2), tw1_stride, vl);
    vint16m1_t v_tw1_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 2) + 1, tw1_stride, vl);
    vint16m1_t v_tw2_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 4), tw2_stride, vl);
    vint16m1_t v_tw2_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 4) + 1, tw2_stride, vl);
    vint16m1_t v_tw3_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 6), tw3_stride, vl);
    vint16m1_t v_tw3_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 6) + 1, tw3_stride, vl);

    // Perform complex multiplications
    vint16m1_t v_s0_r, v_s0_i, v_s1_r, v_s1_i, v_s2_r, v_s2_i;
    do
    {
      vint32m2_t ac = __riscv_vwmul_vv_i32m2(v_f1d_r, v_tw1_r, vl);
      vint32m2_t bd = __riscv_vwmul_vv_i32m2(v_f1d_i, v_tw1_i, vl);
      vint32m2_t ad = __riscv_vwmul_vv_i32m2(v_f1d_r, v_tw1_i, vl);
      vint32m2_t bc = __riscv_vwmul_vv_i32m2(v_f1d_i, v_tw1_r, vl);
      v_s0_r = __riscv_vnclip_wx_i16m1(__riscv_vssra_vx_i32m2(
          __riscv_vsub_vv_i32m2(ac, bd, vl), 15, __RISCV_VXRM_RNU, vl),
                                       0, __RISCV_VXRM_RNU, vl);
      v_s0_i = __riscv_vnclip_wx_i16m1(__riscv_vssra_vx_i32m2(
          __riscv_vadd_vv_i32m2(ad, bc, vl), 15, __RISCV_VXRM_RNU, vl),
                                       0, __RISCV_VXRM_RNU, vl);
    } while (0);

    do
    {
      vint32m2_t ac = __riscv_vwmul_vv_i32m2(v_f2d_r, v_tw2_r, vl);
      vint32m2_t bd = __riscv_vwmul_vv_i32m2(v_f2d_i, v_tw2_i, vl);
      vint32m2_t ad = __riscv_vwmul_vv_i32m2(v_f2d_r, v_tw2_i, vl);
      vint32m2_t bc = __riscv_vwmul_vv_i32m2(v_f2d_i, v_tw2_r, vl);
      v_s1_r = __riscv_vnclip_wx_i16m1(__riscv_vssra_vx_i32m2(
          __riscv_vsub_vv_i32m2(ac, bd, vl), 15, __RISCV_VXRM_RNU, vl),
                                       0, __RISCV_VXRM_RNU, vl);
      v_s1_i = __riscv_vnclip_wx_i16m1(__riscv_vssra_vx_i32m2(
          __riscv_vadd_vv_i32m2(ad, bc, vl), 15, __RISCV_VXRM_RNU, vl),
                                       0, __RISCV_VXRM_RNU, vl);
    } while (0);

    do
    {
      vint32m2_t ac = __riscv_vwmul_vv_i32m2(v_f3d_r, v_tw3_r, vl);
      vint32m2_t bd = __riscv_vwmul_vv_i32m2(v_f3d_i, v_tw3_i, vl);
      vint32m2_t ad = __riscv_vwmul_vv_i32m2(v_f3d_r, v_tw3_i, vl);
      vint32m2_t bc = __riscv_vwmul_vv_i32m2(v_f3d_i, v_tw3_r, vl);
      v_s2_r = __riscv_vnclip_wx_i16m1(__riscv_vssra_vx_i32m2(
          __riscv_vsub_vv_i32m2(ac, bd, vl), 15, __RISCV_VXRM_RNU, vl),
                                       0, __RISCV_VXRM_RNU, vl);
      v_s2_i = __riscv_vnclip_wx_i16m1(__riscv_vssra_vx_i32m2(
          __riscv_vadd_vv_i32m2(ad, bc, vl), 15, __RISCV_VXRM_RNU, vl),
                                       0, __RISCV_VXRM_RNU, vl);
    } while (0);

    // Calculate intermediate butterfly values
    vint16m1_t v_s5_r = __riscv_vsub_vv_i16m1(v_f0d_r, v_s1_r, vl);
    vint16m1_t v_s5_i = __riscv_vsub_vv_i16m1(v_f0d_i, v_s1_i, vl);
    vint16m1_t v_f0d_plus_s1_r = __riscv_vadd_vv_i16m1(v_f0d_r, v_s1_r, vl);
    vint16m1_t v_f0d_plus_s1_i = __riscv_vadd_vv_i16m1(v_f0d_i, v_s1_i, vl);
    vint16m1_t v_s3_r = __riscv_vadd_vv_i16m1(v_s0_r, v_s2_r, vl);
    vint16m1_t v_s3_i = __riscv_vadd_vv_i16m1(v_s0_i, v_s2_i, vl);
    vint16m1_t v_s4_r = __riscv_vsub_vv_i16m1(v_s0_r, v_s2_r, vl);
    vint16m1_t v_s4_i = __riscv_vsub_vv_i16m1(v_s0_i, v_s2_i, vl);
    vint16m1_t v_res_f0_r = __riscv_vadd_vv_i16m1(v_f0d_plus_s1_r, v_s3_r, vl);
    vint16m1_t v_res_f0_i = __riscv_vadd_vv_i16m1(v_f0d_plus_s1_i, v_s3_i, vl);
    vint16m1_t v_res_f2_r = __riscv_vsub_vv_i16m1(v_f0d_plus_s1_r, v_s3_r, vl);
    vint16m1_t v_res_f2_i = __riscv_vsub_vv_i16m1(v_f0d_plus_s1_i, v_s3_i, vl);

    // Calculate final results, handling inverse case
    vint16m1_t v_res_f1_r, v_res_f1_i, v_res_f3_r, v_res_f3_i;
    if (st->inverse)
    {
      v_res_f1_r = __riscv_vsub_vv_i16m1(v_s5_r, v_s4_i, vl);
      v_res_f1_i = __riscv_vadd_vv_i16m1(v_s5_i, v_s4_r, vl);
      v_res_f3_r = __riscv_vadd_vv_i16m1(v_s5_r, v_s4_i, vl);
      v_res_f3_i = __riscv_vsub_vv_i16m1(v_s5_i, v_s4_r, vl);
    }
    else
    {
      v_res_f1_r = __riscv_vadd_vv_i16m1(v_s5_r, v_s4_i, vl);
      v_res_f1_i = __riscv_vsub_vv_i16m1(v_s5_i, v_s4_r, vl);
      v_res_f3_r = __riscv_vsub_vv_i16m1(v_s5_r, v_s4_i, vl);
      v_res_f3_i = __riscv_vadd_vv_i16m1(v_s5_i, v_s4_r, vl);
    }

    // Store final results
    __riscv_vsse16_v_i16m1(Fout0_base + 2 * k, cpx_stride, v_res_f0_r, vl);
    __riscv_vsse16_v_i16m1(Fout0_base + 2 * k + 1, cpx_stride, v_res_f0_i, vl);
    __riscv_vsse16_v_i16m1(Fout1_base + 2 * k, cpx_stride, v_res_f1_r, vl);
    __riscv_vsse16_v_i16m1(Fout1_base + 2 * k + 1, cpx_stride, v_res_f1_i, vl);
    __riscv_vsse16_v_i16m1(Fout2_base + 2 * k, cpx_stride, v_res_f2_r, vl);
    __riscv_vsse16_v_i16m1(Fout2_base + 2 * k + 1, cpx_stride, v_res_f2_i, vl);
    __riscv_vsse16_v_i16m1(Fout3_base + 2 * k, cpx_stride, v_res_f3_r, vl);
    __riscv_vsse16_v_i16m1(Fout3_base + 2 * k + 1, cpx_stride, v_res_f3_i, vl);

    // Advance loop counter
    k += vl;
  }
}

static void kf_bfly3_rvv(kiss_fft_fixed16::kiss_fft_cpx* Fout,
                         const size_t fstride,
                         const kiss_fft_fixed16::kiss_fft_cfg st, size_t m)
{
  // Initialize pointers and constants
  kiss_fft_fixed16::kiss_fft_cpx* Fout1 = Fout + m;
  kiss_fft_fixed16::kiss_fft_cpx* Fout2 = Fout + m * 2;
  const int16_t* tw1_base = (const int16_t*)st->twiddles;
  const int16_t* tw2_base = tw1_base;
  const int16_t tw3i = -28378;  // Q15 value for sin(-2*pi/3)
  int16_t* Fout0_base = (int16_t*)Fout;
  int16_t* Fout1_base = (int16_t*)Fout1;
  int16_t* Fout2_base = (int16_t*)Fout2;
  ptrdiff_t cpx_stride = sizeof(kiss_fft_fixed16::kiss_fft_cpx);
  ptrdiff_t tw1_stride = fstride * cpx_stride;
  ptrdiff_t tw2_stride = fstride * 2 * cpx_stride;

  // Main processing loop
  size_t k = 0;
  while (k < m)
  {
    // Set the vector length for this iteration (LMUL=1)
    size_t vl = __riscv_vsetvl_e16m1(m - k);

    // Load input data vectors
    vint16m1_t v_f0_r =
        __riscv_vlse16_v_i16m1(Fout0_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f0_i =
        __riscv_vlse16_v_i16m1(Fout0_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f1_r =
        __riscv_vlse16_v_i16m1(Fout1_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f1_i =
        __riscv_vlse16_v_i16m1(Fout1_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f2_r =
        __riscv_vlse16_v_i16m1(Fout2_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f2_i =
        __riscv_vlse16_v_i16m1(Fout2_base + 2 * k + 1, cpx_stride, vl);

    // Load twiddle factor vectors
    vint16m1_t v_tw1_r =
        __riscv_vlse16_v_i16m1(tw1_base + (k * fstride * 2), tw1_stride, vl);
    vint16m1_t v_tw1_i =
        __riscv_vlse16_v_i16m1(tw1_base + (k * fstride * 2) + 1, tw1_stride, vl);
    vint16m1_t v_tw2_r =
        __riscv_vlse16_v_i16m1(tw2_base + (k * fstride * 4), tw2_stride, vl);
    vint16m1_t v_tw2_i =
        __riscv_vlse16_v_i16m1(tw2_base + (k * fstride * 4) + 1, tw2_stride, vl);

    // Perform complex multiplications: v_s0 = v_f1 * v_tw1
    vint32m2_t v_ac0 = __riscv_vwmul_vv_i32m2(v_f1_r, v_tw1_r, vl);
    vint32m2_t v_bd0 = __riscv_vwmul_vv_i32m2(v_f1_i, v_tw1_i, vl);
    vint32m2_t v_ad0 = __riscv_vwmul_vv_i32m2(v_f1_r, v_tw1_i, vl);
    vint32m2_t v_bc0 = __riscv_vwmul_vv_i32m2(v_f1_i, v_tw1_r, vl);
    vint16m1_t v_s0_r = __riscv_vnclip_wx_i16m1(
        __riscv_vssra_vx_i32m2(__riscv_vsub_vv_i32m2(v_ac0, v_bd0, vl), 15,
                               __RISCV_VXRM_RNU, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_s0_i = __riscv_vnclip_wx_i16m1(
        __riscv_vssra_vx_i32m2(__riscv_vadd_vv_i32m2(v_ad0, v_bc0, vl), 15,
                               __RISCV_VXRM_RNU, vl),
        0, __RISCV_VXRM_RNU, vl);

    // Perform complex multiplications
    vint32m2_t v_ac1 = __riscv_vwmul_vv_i32m2(v_f2_r, v_tw2_r, vl);
    vint32m2_t v_bd1 = __riscv_vwmul_vv_i32m2(v_f2_i, v_tw2_i, vl);
    vint32m2_t v_ad1 = __riscv_vwmul_vv_i32m2(v_f2_r, v_tw2_i, vl);
    vint32m2_t v_bc1 = __riscv_vwmul_vv_i32m2(v_f2_i, v_tw2_r, vl);
    vint16m1_t v_s1_r = __riscv_vnclip_wx_i16m1(
        __riscv_vssra_vx_i32m2(__riscv_vsub_vv_i32m2(v_ac1, v_bd1, vl), 15,
                               __RISCV_VXRM_RNU, vl),
        0, __RISCV_VXRM_RNU, vl);
    vint16m1_t v_s1_i = __riscv_vnclip_wx_i16m1(
        __riscv_vssra_vx_i32m2(__riscv_vadd_vv_i32m2(v_ad1, v_bc1, vl), 15,
                               __RISCV_VXRM_RNU, vl),
        0, __RISCV_VXRM_RNU, vl);

    // Calculate intermediate butterfly values
    vint16m1_t v_s_add_r = __riscv_vadd_vv_i16m1(v_s0_r, v_s1_r, vl);
    vint16m1_t v_s_add_i = __riscv_vadd_vv_i16m1(v_s0_i, v_s1_i, vl);
    vint16m1_t v_s_sub_r = __riscv_vsub_vv_i16m1(v_s0_r, v_s1_r, vl);
    vint16m1_t v_s_sub_i = __riscv_vsub_vv_i16m1(v_s0_i, v_s1_i, vl);

    // Calculate Fout0 = Fout0 + s_add
    vint16m1_t v_res_f0_r = __riscv_vadd_vv_i16m1(v_f0_r, v_s_add_r, vl);
    vint16m1_t v_res_f0_i = __riscv_vadd_vv_i16m1(v_f0_i, v_s_add_i, vl);

    // Calculate remaining outputs using rotations
    vint16m1_t v_s_add_r_neg_half =
        __riscv_vneg_v_i16m1(__riscv_vsra_vx_i16m1(v_s_add_r, 1, vl), vl);
    vint16m1_t v_s_add_i_neg_half =
        __riscv_vneg_v_i16m1(__riscv_vsra_vx_i16m1(v_s_add_i, 1, vl), vl);
    vint32m2_t v_s_sub_i_mul_tw3i = __riscv_vwmul_vx_i32m2(v_s_sub_i, tw3i, vl);
    vint32m2_t v_s_sub_r_mul_tw3i = __riscv_vwmul_vx_i32m2(v_s_sub_r, tw3i, vl);
    vint16m1_t v_s_sub_i_scaled = __riscv_vnclip_wx_i16m1(
        __riscv_vssra_vx_i32m2(v_s_sub_i_mul_tw3i, 15, __RISCV_VXRM_RNU, vl), 0,
        __RISCV_VXRM_RNU, vl);
    vint16m1_t v_s_sub_r_scaled = __riscv_vnclip_wx_i16m1(
        __riscv_vssra_vx_i32m2(v_s_sub_r_mul_tw3i, 15, __RISCV_VXRM_RNU, vl), 0,
        __RISCV_VXRM_RNU, vl);
    vint16m1_t v_tmp_r1 = __riscv_vadd_vv_i16m1(v_f0_r, v_s_add_r_neg_half, vl);
    vint16m1_t v_res_f1_r = __riscv_vsub_vv_i16m1(v_tmp_r1, v_s_sub_i_scaled, vl);
    vint16m1_t v_tmp_i1 = __riscv_vadd_vv_i16m1(v_f0_i, v_s_add_i_neg_half, vl);
    vint16m1_t v_res_f1_i = __riscv_vadd_vv_i16m1(v_tmp_i1, v_s_sub_r_scaled, vl);
    vint16m1_t v_res_f2_r = __riscv_vadd_vv_i16m1(v_tmp_r1, v_s_sub_i_scaled, vl);
    vint16m1_t v_res_f2_i = __riscv_vsub_vv_i16m1(v_tmp_i1, v_s_sub_r_scaled, vl);

    // Store results
    __riscv_vsse16_v_i16m1(Fout0_base + 2 * k, cpx_stride, v_res_f0_r, vl);
    __riscv_vsse16_v_i16m1(Fout0_base + 2 * k + 1, cpx_stride, v_res_f0_i, vl);
    __riscv_vsse16_v_i16m1(Fout1_base + 2 * k, cpx_stride, v_res_f1_r, vl);
    __riscv_vsse16_v_i16m1(Fout1_base + 2 * k + 1, cpx_stride, v_res_f1_i, vl);
    __riscv_vsse16_v_i16m1(Fout2_base + 2 * k, cpx_stride, v_res_f2_r, vl);
    __riscv_vsse16_v_i16m1(Fout2_base + 2 * k + 1, cpx_stride, v_res_f2_i, vl);

    // Advance loop counter
    k += vl;
  }
}

static void kf_bfly5_rvv(kiss_fft_fixed16::kiss_fft_cpx* Fout,
                         const size_t fstride,
                         const kiss_fft_fixed16::kiss_fft_cfg st, size_t m)
{
  // Initialize pointers and constants
  kiss_fft_fixed16::kiss_fft_cpx *Fout0, *Fout1, *Fout2, *Fout3, *Fout4;
  const int16_t* tw_base = (const int16_t*)st->twiddles;
  const int16_t ya1 = 19021;   // Q15 value for cos(2*pi/5)
  const int16_t yb1 = 31164;   // Q15 value for sin(2*pi/5)
  const int16_t ya2 = -30777;  // Q15 value for cos(4*pi/5)
  const int16_t yb2 = 19021;   // Q15 value for sin(4*pi/5)

  Fout0 = Fout;
  Fout1 = Fout + m;
  Fout2 = Fout + 2 * m;
  Fout3 = Fout + 3 * m;
  Fout4 = Fout + 4 * m;

  int16_t* Fout0_base = (int16_t*)Fout0;
  int16_t* Fout1_base = (int16_t*)Fout1;
  int16_t* Fout2_base = (int16_t*)Fout2;
  int16_t* Fout3_base = (int16_t*)Fout3;
  int16_t* Fout4_base = (int16_t*)Fout4;

  ptrdiff_t cpx_stride = sizeof(kiss_fft_fixed16::kiss_fft_cpx);
  ptrdiff_t tw1_stride = fstride * cpx_stride;
  ptrdiff_t tw2_stride = 2 * tw1_stride;
  ptrdiff_t tw3_stride = 3 * tw1_stride;
  ptrdiff_t tw4_stride = 4 * tw1_stride;

  // Main processing loop
  size_t k = 0;
  while (k < m)
  {
    // Set the vector length for this iteration
    size_t vl = __riscv_vsetvl_e16m1(m - k);

    // Load input data vectors
    vint16m1_t v_f0_r =
        __riscv_vlse16_v_i16m1(Fout0_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f0_i =
        __riscv_vlse16_v_i16m1(Fout0_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f1_r =
        __riscv_vlse16_v_i16m1(Fout1_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f1_i =
        __riscv_vlse16_v_i16m1(Fout1_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f2_r =
        __riscv_vlse16_v_i16m1(Fout2_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f2_i =
        __riscv_vlse16_v_i16m1(Fout2_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f3_r =
        __riscv_vlse16_v_i16m1(Fout3_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f3_i =
        __riscv_vlse16_v_i16m1(Fout3_base + 2 * k + 1, cpx_stride, vl);
    vint16m1_t v_f4_r =
        __riscv_vlse16_v_i16m1(Fout4_base + 2 * k, cpx_stride, vl);
    vint16m1_t v_f4_i =
        __riscv_vlse16_v_i16m1(Fout4_base + 2 * k + 1, cpx_stride, vl);

    // Load twiddle factor vectors
    vint16m1_t v_tw1_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 2), tw1_stride, vl);
    vint16m1_t v_tw1_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 2) + 1, tw1_stride, vl);
    vint16m1_t v_tw2_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 4), tw2_stride, vl);
    vint16m1_t v_tw2_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 4) + 1, tw2_stride, vl);
    vint16m1_t v_tw3_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 6), tw3_stride, vl);
    vint16m1_t v_tw3_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 6) + 1, tw3_stride, vl);
    vint16m1_t v_tw4_r =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 8), tw4_stride, vl);
    vint16m1_t v_tw4_i =
        __riscv_vlse16_v_i16m1(tw_base + (k * fstride * 8) + 1, tw4_stride, vl);

// Macro for complex multiplication, wrapped in do-while(0) to prevent scope issues
#define C_MUL_VEC(res_r, res_i, f_r, f_i, tw_r, tw_i)                           \
  do                                                                           \
  {                                                                            \
    vint32m2_t ac = __riscv_vwmul_vv_i32m2(f_r, tw_r, vl);                      \
    vint32m2_t bd = __riscv_vwmul_vv_i32m2(f_i, tw_i, vl);                      \
    vint32m2_t ad = __riscv_vwmul_vv_i32m2(f_r, tw_i, vl);                      \
    vint32m2_t bc = __riscv_vwmul_vv_i32m2(f_i, tw_r, vl);                      \
    res_r = __riscv_vnclip_wx_i16m1(                                           \
        __riscv_vssra_vx_i32m2(__riscv_vsub_vv_i32m2(ac, bd, vl), 15,           \
                               __RISCV_VXRM_RNU, vl),                          \
        0, __RISCV_VXRM_RNU, vl);                                              \
    res_i = __riscv_vnclip_wx_i16m1(                                           \
        __riscv_vssra_vx_i32m2(__riscv_vadd_vv_i32m2(ad, bc, vl), 15,           \
                               __RISCV_VXRM_RNU, vl),                          \
        0, __RISCV_VXRM_RNU, vl);                                              \
  } while (0)

    // Perform complex multiplications
    vint16m1_t v_s0_r, v_s0_i, v_s1_r, v_s1_i, v_s2_r, v_s2_i, v_s3_r, v_s3_i;
    C_MUL_VEC(v_s0_r, v_s0_i, v_f1_r, v_f1_i, v_tw1_r, v_tw1_i);
    C_MUL_VEC(v_s1_r, v_s1_i, v_f2_r, v_f2_i, v_tw2_r, v_tw2_i);
    C_MUL_VEC(v_s2_r, v_s2_i, v_f3_r, v_f3_i, v_tw3_r, v_tw3_i);
    C_MUL_VEC(v_s3_r, v_s3_i, v_f4_r, v_f4_i, v_tw4_r, v_tw4_i);
#undef C_MUL_VEC

    // Calculate intermediate butterfly values
    vint16m1_t v_s03_add_r = __riscv_vadd_vv_i16m1(v_s0_r, v_s3_r, vl);
    vint16m1_t v_s03_add_i = __riscv_vadd_vv_i16m1(v_s0_i, v_s3_i, vl);
    vint16m1_t v_s03_sub_r = __riscv_vsub_vv_i16m1(v_s0_r, v_s3_r, vl);
    vint16m1_t v_s03_sub_i = __riscv_vsub_vv_i16m1(v_s0_i, v_s3_i, vl);
    vint16m1_t v_s12_add_r = __riscv_vadd_vv_i16m1(v_s1_r, v_s2_r, vl);
    vint16m1_t v_s12_add_i = __riscv_vadd_vv_i16m1(v_s1_i, v_s2_i, vl);
    vint16m1_t v_s12_sub_r = __riscv_vsub_vv_i16m1(v_s1_r, v_s2_r, vl);
    vint16m1_t v_s12_sub_i = __riscv_vsub_vv_i16m1(v_s1_i, v_s2_i, vl);

    // Calculate Fout0 = f0 + s03_add + s12_add
    vint16m1_t v_res_f0_r = __riscv_vadd_vv_i16m1(
        v_f0_r, __riscv_vadd_vv_i16m1(v_s03_add_r, v_s12_add_r, vl), vl);
    vint16m1_t v_res_f0_i = __riscv_vadd_vv_i16m1(
        v_f0_i, __riscv_vadd_vv_i16m1(v_s03_add_i, v_s12_add_i, vl), vl);

// Macro for scalar multiplication, wrapped in do-while(0) to prevent scope issues
#define S_MUL_VX(res, val, const_val)                                          \
  do                                                                           \
  {                                                                            \
    vint32m2_t tmp_mul = __riscv_vwmul_vx_i32m2(val, const_val, vl);            \
    res = __riscv_vnclip_wx_i16m1(                                             \
        __riscv_vssra_vx_i32m2(tmp_mul, 15, __RISCV_VXRM_RNU, vl), 0,           \
        __RISCV_VXRM_RNU, vl);                                                 \
  } while (0)

    // Perform final rotations
    vint16m1_t v_tmp1_r, v_tmp1_i, v_tmp2_r, v_tmp2_i;
    S_MUL_VX(v_tmp1_r, v_s03_add_r, ya1);
    S_MUL_VX(v_tmp1_i, v_s03_add_i, ya1);
    S_MUL_VX(v_tmp2_r, v_s12_add_r, ya2);
    S_MUL_VX(v_tmp2_i, v_s12_add_i, ya2);
    vint16m1_t v_r_part1 = __riscv_vadd_vv_i16m1(
        v_f0_r, __riscv_vadd_vv_i16m1(v_tmp1_r, v_tmp2_r, vl), vl);
    vint16m1_t v_i_part1 = __riscv_vadd_vv_i16m1(
        v_f0_i, __riscv_vadd_vv_i16m1(v_tmp1_i, v_tmp2_i, vl), vl);
    S_MUL_VX(v_tmp1_r, v_s03_sub_i, yb1);
    S_MUL_VX(v_tmp1_i, v_s03_sub_r, yb1);
    S_MUL_VX(v_tmp2_r, v_s12_sub_i, yb2);
    S_MUL_VX(v_tmp2_i, v_s12_sub_r, yb2);
    vint16m1_t v_r_part2 = __riscv_vsub_vv_i16m1(v_tmp1_r, v_tmp2_r, vl);
    vint16m1_t v_i_part2 = __riscv_vadd_vv_i16m1(v_tmp1_i, v_tmp2_i, vl);

    // Calculate final butterfly outputs
    vint16m1_t v_res_f1_r = __riscv_vadd_vv_i16m1(v_r_part1, v_r_part2, vl);
    vint16m1_t v_res_f1_i = __riscv_vadd_vv_i16m1(v_i_part1, v_i_part2, vl);
    vint16m1_t v_res_f4_r = __riscv_vsub_vv_i16m1(v_r_part1, v_r_part2, vl);
    vint16m1_t v_res_f4_i = __riscv_vsub_vv_i16m1(v_i_part1, v_i_part2, vl);
    v_r_part2 = __riscv_vadd_vv_i16m1(v_tmp1_r, v_tmp2_r, vl);
    v_i_part2 = __riscv_vsub_vv_i16m1(v_tmp1_i, v_tmp2_i, vl);
    vint16m1_t v_res_f2_r = __riscv_vsub_vv_i16m1(v_r_part1, v_r_part2, vl);
    vint16m1_t v_res_f2_i = __riscv_vadd_vv_i16m1(v_i_part1, v_i_part2, vl);
    vint16m1_t v_res_f3_r = __riscv_vadd_vv_i16m1(v_r_part1, v_r_part2, vl);
    vint16m1_t v_res_f3_i = __riscv_vsub_vv_i16m1(v_i_part1, v_i_part2, vl);
#undef S_MUL_VX

    // Store results
    __riscv_vsse16_v_i16m1(Fout0_base + 2 * k, cpx_stride, v_res_f0_r, vl);
    __riscv_vsse16_v_i16m1(Fout0_base + 2 * k + 1, cpx_stride, v_res_f0_i, vl);
    __riscv_vsse16_v_i16m1(Fout1_base + 2 * k, cpx_stride, v_res_f1_r, vl);
    __riscv_vsse16_v_i16m1(Fout1_base + 2 * k + 1, cpx_stride, v_res_f1_i, vl);
    __riscv_vsse16_v_i16m1(Fout2_base + 2 * k, cpx_stride, v_res_f2_r, vl);
    __riscv_vsse16_v_i16m1(Fout2_base + 2 * k + 1, cpx_stride, v_res_f2_i, vl);
    __riscv_vsse16_v_i16m1(Fout3_base + 2 * k, cpx_stride, v_res_f3_r, vl);
    __riscv_vsse16_v_i16m1(Fout3_base + 2 * k + 1, cpx_stride, v_res_f3_i, vl);
    __riscv_vsse16_v_i16m1(Fout4_base + 2 * k, cpx_stride, v_res_f4_r, vl);
    __riscv_vsse16_v_i16m1(Fout4_base + 2 * k + 1, cpx_stride, v_res_f4_i, vl);

    // Advance loop counter
    k += vl;
  }
}

// Generic radix implementation copy/pasted from kissfft (kiss_fft.c)
static void kf_bfly_generic(
        kiss_fft_fixed16::kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_fixed16::kiss_fft_cfg st,
        int m,
        int p
        )
{
    int u,k,q1,q;
    kiss_fft_fixed16::kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_fixed16::kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_fixed16::kiss_fft_cpx * scratch = (kiss_fft_fixed16::kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(sizeof(kiss_fft_fixed16::kiss_fft_cpx)*p);
    if (scratch == NULL){
        return;
    }

    for ( u=0; u<m; ++u ) {
        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            scratch[q1] = Fout[ k  ];
            C_FIXDIV(scratch[q1],p);
            k += m;
        }

        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            int twidx=0;
            Fout[ k ] = scratch[0];
            for (q=1;q<p;++q ) {
                twidx += fstride * k;
                if (twidx>=Norig) twidx-=Norig;
                C_MUL(t,scratch[q] , twiddles[twidx] );
                C_ADDTO( Fout[ k ] ,t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}

static void kf_work_rvv(kiss_fft_fixed16::kiss_fft_cpx* Fout,
                        const kiss_fft_fixed16::kiss_fft_cpx* f,
                        const size_t fstride, int in_stride, int* factors,
                        const kiss_fft_fixed16::kiss_fft_cfg st)
{
  // Decompose the problem into factors p and m
  const int p = *factors++;
  const int m = *factors++;
  kiss_fft_fixed16::kiss_fft_cpx* Fout_beg = Fout;
  const kiss_fft_fixed16::kiss_fft_cpx* Fout_end = Fout + p * m;

  // Perform recursion for the m-point DFTs
  if (m == 1)
  {
    do
    {
      *Fout = *f;
      f += fstride * in_stride;
    } while (++Fout != Fout_end);
  }
  else
  {
    do
    {
      kf_work_rvv(Fout, f, fstride * p, in_stride, factors, st);
      f += fstride * in_stride;
    } while ((Fout += m) != Fout_end);
  }

  // Perform the p-point butterfly operations
  Fout = Fout_beg;
  switch (p)
  {
    case 2:
      kf_bfly2_rvv(Fout, fstride, st, m);
      break;
    case 3:
      kf_bfly3_rvv(Fout, fstride, st, m);
      break;
    case 4:
      kf_bfly4_rvv(Fout, fstride, st, m);
      break;
    case 5:
      kf_bfly5_rvv(Fout, fstride, st, m);
      break;
      default: kf_bfly_generic(Fout, fstride, st, m, p); break;
  }
}

void kiss_fft_stride_rvv(kiss_fft_fixed16::kiss_fft_cfg st, const kiss_fft_fixed16::kiss_fft_cpx* fin,
                         kiss_fft_fixed16::kiss_fft_cpx* fout, int in_stride)
{
  // Handle in-place transform
  if (fin == fout)
  {
    if (fout == NULL)
    {
      return;
    }

    kiss_fft_fixed16::kiss_fft_cpx* tmpbuf =
        (kiss_fft_fixed16::kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(
            sizeof(kiss_fft_fixed16::kiss_fft_cpx) * st->nfft);

    if (tmpbuf == NULL)
    {
      return;
    }

    kf_work_rvv(tmpbuf, fin, 1, in_stride, st->factors, st);

    memcpy(fout, tmpbuf, sizeof(kiss_fft_fixed16::kiss_fft_cpx) * st->nfft);

    KISS_FFT_TMP_FREE(tmpbuf);
  }
  else
  {
    // Handle out-of-place transform
    kf_work_rvv(fout, fin, 1, in_stride, st->factors, st);
  }
}

void kiss_fft_rvv(kiss_fft_fixed16::kiss_fft_cfg cfg, const kiss_fft_fixed16::kiss_fft_cpx* fin, kiss_fft_fixed16::kiss_fft_cpx* fout)
{
  kiss_fft_stride_rvv(cfg, fin, fout, 1);
}

void kiss_fftr_rvv(kiss_fft_fixed16::kiss_fftr_cfg st, const kiss_fft_scalar* timedata,
                   kiss_fft_fixed16::kiss_fft_cpx* freqdata)
{
  // Handle inverse FFT case and perform the initial complex FFT
  if (st->substate->inverse)
  {
    return;
  }
  kiss_fft_rvv(st->substate, (const kiss_fft_fixed16::kiss_fft_cpx*)timedata, st->tmpbuf);

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

  // Initialize pointers and loop variables
  size_t k = 1;
  const size_t loop_end = ncfft / 2;
  const int16_t* tmpbuf_base_ptr = (const int16_t*)st->tmpbuf;
  const int16_t* twiddles_base_ptr = (const int16_t*)st->super_twiddles;
  int16_t* freqdata_base_ptr = (int16_t*)freqdata;
  
  // Stride for complex numbers (R, I) is 4 bytes (2 * int16)
  ptrdiff_t stride = sizeof(kiss_fft_fixed16::kiss_fft_cpx);
  ptrdiff_t neg_stride = -stride;

  // Main loop to process FFT bins in vector chunks
  while (k <= loop_end)
  {
    // Set the vector length (vl) for the current iteration
    // Optimization: Reduced to m2 to prevent register spilling
    size_t vl = __riscv_vsetvl_e16m2(loop_end - k + 1);

    // fpk indices: k, k+1, ...
    vint16m2_t v_fpk_r = __riscv_vlse16_v_i16m2(&tmpbuf_base_ptr[2 * k], stride, vl);
    vint16m2_t v_fpk_i = __riscv_vlse16_v_i16m2(&tmpbuf_base_ptr[2 * k + 1], stride, vl);

    // fpnk indices: N-k, N-(k+1), ...
    const int16_t* fpnk_ptr = &tmpbuf_base_ptr[2 * (ncfft - k)];
    vint16m2_t v_fpnk_r_raw = __riscv_vlse16_v_i16m2(fpnk_ptr, neg_stride, vl);
    vint16m2_t v_fpnk_i_raw = __riscv_vlse16_v_i16m2(fpnk_ptr + 1, neg_stride, vl);

    // Twiddle indices: k-1, k, ...
    // Must use strided load to extract only Reals or only Imags from the interleaved array
    const int16_t* tw_ptr = &twiddles_base_ptr[2 * (k - 1)];
    vint16m2_t v_tw_r = __riscv_vlse16_v_i16m2(tw_ptr, stride, vl);
    vint16m2_t v_tw_i = __riscv_vlse16_v_i16m2(tw_ptr + 1, stride, vl);

    // Perform high-precision rounding division on fpk
    const int16_t scale = 16383;
    const int32_t round_const = 16384;
    vint32m4_t v_fpk_r_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fpk_r, scale, vl), round_const, vl), 15, vl);
    vint32m4_t v_fpk_i_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fpk_i, scale, vl), round_const, vl), 15, vl);
    vint16m2_t v_fpk_r_div2 = __riscv_vnclip_wx_i16m2(v_fpk_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_fpk_i_div2 = __riscv_vnclip_wx_i16m2(v_fpk_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Perform high-precision rounding division on fpnk (with negated imaginary part)
    vint16m2_t v_fpnk_i_neg = __riscv_vneg_v_i16m2(v_fpnk_i_raw, vl);
    vint32m4_t v_fpnk_r_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fpnk_r_raw, scale, vl), round_const, vl), 15, vl);
    vint32m4_t v_fpnk_i_32 = __riscv_vsra_vx_i32m4(
        __riscv_vadd_vx_i32m4(__riscv_vwmul_vx_i32m4(v_fpnk_i_neg, scale, vl), round_const, vl), 15, vl);
    vint16m2_t v_fpnk_r_div2 = __riscv_vnclip_wx_i16m2(v_fpnk_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_fpnk_i_div2 = __riscv_vnclip_wx_i16m2(v_fpnk_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Calculate intermediate values f1k (add) and f2k (subtract)
    vint16m2_t v_f1k_r = __riscv_vadd_vv_i16m2(v_fpk_r_div2, v_fpnk_r_div2, vl);
    vint16m2_t v_f1k_i = __riscv_vadd_vv_i16m2(v_fpk_i_div2, v_fpnk_i_div2, vl);
    vint16m2_t v_f2k_r = __riscv_vsub_vv_i16m2(v_fpk_r_div2, v_fpnk_r_div2, vl);
    vint16m2_t v_f2k_i = __riscv_vsub_vv_i16m2(v_fpk_i_div2, v_fpnk_i_div2, vl);

    // Perform complex multiplication
    vint32m4_t v_ac = __riscv_vwmul_vv_i32m4(v_f2k_r, v_tw_r, vl);
    vint32m4_t v_bd = __riscv_vwmul_vv_i32m4(v_f2k_i, v_tw_i, vl);
    vint32m4_t v_ad = __riscv_vwmul_vv_i32m4(v_f2k_r, v_tw_i, vl);
    vint32m4_t v_bc = __riscv_vwmul_vv_i32m4(v_f2k_i, v_tw_r, vl);
    vint32m4_t v_tw_res_r_32 = __riscv_vssra_vx_i32m4(__riscv_vsub_vv_i32m4(v_ac, v_bd, vl), 15, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_tw_res_i_32 = __riscv_vssra_vx_i32m4(__riscv_vadd_vv_i32m4(v_ad, v_bc, vl), 15, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_tw_res_r = __riscv_vnclip_wx_i16m2(v_tw_res_r_32, 0, __RISCV_VXRM_RNU, vl);
    vint16m2_t v_tw_res_i = __riscv_vnclip_wx_i16m2(v_tw_res_i_32, 0, __RISCV_VXRM_RNU, vl);

    // Calculate final output vectors
    vint16m2_t v_out_k_r = __riscv_vsra_vx_i16m2(__riscv_vadd_vv_i16m2(v_f1k_r, v_tw_res_r, vl), 1, vl);
    vint16m2_t v_out_k_i = __riscv_vsra_vx_i16m2(__riscv_vadd_vv_i16m2(v_f1k_i, v_tw_res_i, vl), 1, vl);
    vint16m2_t v_out_nk_r = __riscv_vsra_vx_i16m2(__riscv_vsub_vv_i16m2(v_f1k_r, v_tw_res_r, vl), 1, vl);
    vint16m2_t v_out_nk_i = __riscv_vsra_vx_i16m2(__riscv_vsub_vv_i16m2(v_tw_res_i, v_f1k_i, vl), 1, vl);

    // Store the results using a strided store (Forward)
    __riscv_vsse16_v_i16m2(&freqdata_base_ptr[2 * k], stride, v_out_k_r, vl);
    __riscv_vsse16_v_i16m2(&freqdata_base_ptr[2 * k + 1], stride, v_out_k_i, vl);

    // Store the results using a strided store (Reverse)
    int16_t* out_nk_ptr = &freqdata_base_ptr[2 * (ncfft - k)];
    __riscv_vsse16_v_i16m2(out_nk_ptr, neg_stride, v_out_nk_r, vl);
    __riscv_vsse16_v_i16m2(out_nk_ptr + 1, neg_stride, v_out_nk_i, vl);

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