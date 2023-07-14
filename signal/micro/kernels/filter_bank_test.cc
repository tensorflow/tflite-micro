/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "signal/micro/kernels/filter_bank_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

TfLiteStatus TestFilterBank(int* input1_dims_data, const uint32_t* input1_data,
                            int* input2_dims_data, const int16_t* input2_data,
                            int* input3_dims_data, const int16_t* input3_data,
                            int* input4_dims_data, const int16_t* input4_data,
                            int* input5_dims_data, const int16_t* input5_data,
                            int* input6_dims_data, const int16_t* input6_data,
                            int* output_dims_data, const uint64_t* golden,
                            const uint8_t* flexbuffers_data,
                            const int flexbuffers_data_len,
                            uint64_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* input3_dims = IntArrayFromInts(input3_dims_data);
  TfLiteIntArray* input4_dims = IntArrayFromInts(input4_dims_data);
  TfLiteIntArray* input5_dims = IntArrayFromInts(input5_dims_data);
  TfLiteIntArray* input6_dims = IntArrayFromInts(input6_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int kInputsSize = 6;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(input3_data, input3_dims),
      CreateTensor(input4_data, input4_dims),
      CreateTensor(input5_data, input5_dims),
      CreateTensor(input6_data, input6_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {6, 0, 1, 2, 3, 4, 5};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 6};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const int output_len = ElementCount(*output_dims);

  TFLMRegistration* registration = tflite::tflm_signal::Register_FILTER_BANK();
  micro::KernelRunner runner(*registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare(
      reinterpret_cast<const char*>(flexbuffers_data), flexbuffers_data_len));

  TF_LITE_ENSURE_STATUS(runner.Invoke());

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FilterBankTest32Channel) {
  int input1_shape[] = {1, 257};
  int input2_shape[] = {1, 117};
  int input3_shape[] = {1, 117};
  int input4_shape[] = {1, 33};
  int input5_shape[] = {1, 33};
  int input6_shape[] = {1, 33};
  int output_shape[] = {1, 32};

  uint64_t output[32];

  const uint32_t input1[] = {
      65451,      11468838,   4280615122, 4283105055, 30080683,   969970,
      1168164,    192770,     344209,     1811809,    1740724,    586130,
      305045,     17981,      169273,     103321,     85277,      529901,
      524660,     116609,     29653,      64345,      13121,      273956,
      593748,     463432,     348169,     77545,      2117,       19277,
      13837,      85,         16322,      1325,       69584,      233930,
      253273,     94180,      8642,       104245,     151937,     231970,
      90405,      95849,      106285,     81938,      76226,      103337,
      303250,     337705,     75140,      43874,      33730,      44761,
      117608,     57322,      9945,       19816,      48674,      19465,
      15696,      52229,      103738,     102541,     126421,     133157,
      33680,      7738,       45029,      57122,      61605,      60138,
      26170,      41444,      210994,     238338,     74324,      21460,
      33125,      3940,       15481,      7709,       24929,      17714,
      170993,     91978,      45965,      214133,     96832,      1800,
      16717,      42341,      87421,      114341,     65161,      26260,
      135077,     245000,     122117,     81188,      107753,     74125,
      86432,      91460,      29648,      2069,       3161,       5002,
      784,        1152,       1424,       277,        452,        2696,
      3610,       2120,       2617,       562,        1153,       4610,
      2906,       65,         786450,     4293722107, 0,          393208,
      2,          196608,     65539,      65537,      4294967295, 65537,
      4294901762, 65535,      4294770689, 65533,      131073,     4294901761,
      131071,     131071,     65535,      4294901764, 4294967295, 0,
      4294901758, 4294901761, 196607,     4294836224, 131070,     4294901762,
      4294901759, 196608,     4294901761, 131071,     131070,     65538,
      0,          4294901761, 65536,      4294836225, 65536,      4294836225,
      4294901757, 65535,      4294901760, 196607,     4294967295, 0,
      131071,     4294901762, 4294836221, 196608,     65536,      1,
      131074,     4294770690, 4294967291, 196611,     4294770687, 262143,
      4294901759, 131071,     1,          4294901759, 196607,     4294705153,
      196607,     4294967294, 65536,      1,          4294901759, 65536,
      0,          65536,      65537,      4294901759, 65536,      3,
      4294836222, 65534,      65536,      65538,      4294836225, 4294901760,
      4294901761, 4294967293, 0,          65534,      131070,     65537,
      4294901762, 65536,      2,          4294836224, 1,          4294901760,
      0,          4294967294, 131073,     4294901760, 65535,      131073,
      4294836224, 65536,      4294901760, 4294901760, 4294967295, 4294901761,
      131071,     4294901760, 131071,     4294836224, 2,          4294901758,
      4294967292, 131073,     0,          65535,      0,          4294901760,
      4294967295, 131073,     4294901764, 4294836223, 4294967295, 65535,
      65537,      65533,      3,          131072,     4294836224, 65537,
      1,          4294967293, 196611,     4294901759, 1};

  const int16_t input2[] = {
      1133, 2373, 3712, 1047, 2564, 66,   1740, 3486, 1202, 3079, 919,  2913,
      865,  2964, 1015, 3210, 1352, 3633, 1859, 123,  2520, 856,  3323, 1726,
      161,  2722, 1215, 3833, 2382, 956,  3652, 2276, 923,  3689, 2380, 1093,
      3922, 2676, 1448, 239,  3144, 1970, 814,  3770, 2646, 1538, 445,  3463,
      2399, 1349, 313,  3386, 2376, 1379, 394,  3517, 2556, 1607, 668,  3837,
      2920, 2013, 1117, 231,  3450, 2583, 1725, 877,  37,   3302, 2480, 1666,
      861,  63,   3369, 2588, 1813, 1046, 287,  3630, 2885, 2147, 1415, 690,
      4067, 3355, 2650, 1950, 1257, 569,  3984, 3308, 2638, 1973, 1314, 661,
      12,   3465, 2827, 2194, 1566, 943,  325,  3808, 3199, 2595, 1996, 1401,
      810,  224,  3738, 3160, 2586, 2017, 1451, 890,  332};

  const int16_t input3[] = {
      2962, 1722, 383,  3048, 1531, 4029, 2355, 609,  2893, 1016, 3176, 1182,
      3230, 1131, 3080, 885,  2743, 462,  2236, 3972, 1575, 3239, 772,  2369,
      3934, 1373, 2880, 262,  1713, 3139, 443,  1819, 3172, 406,  1715, 3002,
      173,  1419, 2647, 3856, 951,  2125, 3281, 325,  1449, 2557, 3650, 632,
      1696, 2746, 3782, 709,  1719, 2716, 3701, 578,  1539, 2488, 3427, 258,
      1175, 2082, 2978, 3864, 645,  1512, 2370, 3218, 4058, 793,  1615, 2429,
      3234, 4032, 726,  1507, 2282, 3049, 3808, 465,  1210, 1948, 2680, 3405,
      28,   740,  1445, 2145, 2838, 3526, 111,  787,  1457, 2122, 2781, 3434,
      4083, 630,  1268, 1901, 2529, 3152, 3770, 287,  896,  1500, 2099, 2694,
      3285, 3871, 357,  935,  1509, 2078, 2644, 3205, 3763};

  const int16_t input4[] = {5,  6,  7,  9,  11, 12, 14, 16, 18,  20,  22,
                            25, 27, 30, 32, 35, 38, 41, 45, 48,  52,  56,
                            60, 64, 69, 74, 79, 84, 89, 95, 102, 108, 115};

  const int16_t input5[] = {0,  1,  2,  4,  6,  7,  9,  11, 13, 15,  17,
                            20, 22, 25, 27, 30, 33, 36, 40, 43, 47,  51,
                            55, 59, 64, 69, 74, 79, 84, 90, 97, 103, 110};

  const int16_t input6[] = {1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3,
                            4, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 7, 6, 7, 7};

  const uint64_t golden[] = {
      5645104312, 3087527471, 5883346002, 10807122775, 2465336182, 853935004,
      1206905130, 3485828019, 1134726750, 832725041,   4442875878, 2122064365,
      178483220,  151483681,  1742660113, 1309124116,  1954305288, 1323857378,
      2750861165, 1340947482, 792522630,  669257768,   1659699572, 940652856,
      1957080469, 1034203505, 1541805928, 1710818326,  2432875876, 2254716277,
      275382345,  57293224};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBank(
          input1_shape, input1, input2_shape, input2, input3_shape, input3,
          input4_shape, input4, input5_shape, input5, input6_shape, input6,
          output_shape, golden, g_gen_data_filter_bank_32_channel,
          g_gen_data_size_filter_bank_32_channel, output));
}

TF_LITE_MICRO_TEST(FilterBankTest16Channel) {
  int input1_shape[] = {1, 129};
  int input2_shape[] = {1, 59};
  int input3_shape[] = {1, 59};
  int input4_shape[] = {1, 17};
  int input5_shape[] = {1, 17};
  int input6_shape[] = {1, 17};
  int output_shape[] = {1, 16};

  uint64_t output[16];

  const uint32_t input1[] = {
      645050, 4644,  3653,  24262, 56660, 43260, 50584, 57902, 31702, 5401,
      45555,  34852, 8518,  43556, 13358, 19350, 40221, 18017, 27284, 64491,
      60099,  17863, 11001, 29076, 32666, 65268, 50947, 28694, 32377, 30014,
      25607,  22547, 45086, 10654, 46797, 8622,  47348, 43085, 5747,  51544,
      50364,  6208,  20696, 59782, 14429, 60125, 37079, 32673, 63457, 60142,
      34042,  11280, 1874,  33734, 62118, 13766, 54398, 47818, 50976, 46930,
      25906,  59441, 25958, 59136, 1756,  18652, 29213, 13379, 51845, 1207,
      55626,  27108, 43771, 35236, 3374,  40959, 47707, 41540, 34282, 27094,
      36329,  13593, 65257, 47006, 46857, 1114,  37106, 18738, 25969, 15461,
      2842,   36470, 32489, 61622, 23613, 29624, 32820, 30438, 9543,  6767,
      23037,  52896, 12059, 32264, 11575, 42400, 43344, 27511, 16712, 6877,
      4910,   50047, 61569, 57237, 48558, 2310,  22192, 7874,  46141, 64056,
      61997,  7298,  31372, 25316, 683,   58940, 18755, 17898, 19196};

  const int16_t input2[] = {
      -2210, 1711, 3237, 1247, 2507, 61,   1019, 899,  206,  146,  2849, 2756,
      1260,  1280, 1951, 213,  617,  2047, 211,  347,  2821, 3747, 150,  1924,
      3962,  942,  1430, 2678, 993,  308,  3364, 2491, 954,  1308, 879,  3950,
      1,     3556, 3628, 2104, 78,   1298, 1080, 342,  1337, 1639, 2352, 829,
      1358,  2498, 1647, 2507, 3816, 3767, 3735, 1155, 2221, 2196, 1160};

  const int16_t input3[] = {
      408,  3574, 1880, 2561, 2011, 3394, 1019, 445,  3901, 343,  1874, 3846,
      3566, 1830, 327,  111,  623,  1037, 2803, 1947, 1518, 661,  3239, 2351,
      1257, 269,  1574, 3431, 3972, 2487, 2181, 1458, 552,  717,  679,  1031,
      1738, 1782, 128,  2242, 353,  1460, 3305, 1424, 3813, 2895, 164,  272,
      3886, 3135, 141,  747,  3233, 1478, 2612, 3837, 3271, 73,   1746};

  const int16_t input4[] = {5,  6,  7,  9,  11, 12, 14, 16, 18,
                            20, 22, 25, 27, 30, 32, 35, 33};

  const int16_t input5[] = {0,  1,  2,  4,  6,  7,  9,  11, 13,
                            15, 17, 20, 22, 25, 27, 30, 33};

  const int16_t input6[] = {1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3};

  const uint64_t golden[] = {104199304, 407748384, 206363744, 200989269,
                             52144406,  230780884, 174394190, 379684049,
                             94840835,  57788823,  531528204, 318265707,
                             263149795, 188110467, 501443259, 200747781};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBank(
          input1_shape, input1, input2_shape, input2, input3_shape, input3,
          input4_shape, input4, input5_shape, input5, input6_shape, input6,
          output_shape, golden, g_gen_data_filter_bank_16_channel,
          g_gen_data_size_filter_bank_16_channel, output));
}

TF_LITE_MICRO_TESTS_END
