# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for filter bank ops."""
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import filter_bank_ops
from tflite_micro.python.tflite_micro.signal.utils import util


class FilterBankOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def testFilterBankCenterFreq(self):
    center_freq = filter_bank_ops._calc_center_freq(41, 125, 7800)
    center_freq_exp = [
        249.2831420898437500, 313.3967285156250000, 377.5103454589843750,
        441.6239624023437500, 505.7375488281250000, 569.8511352539062500,
        633.9647827148437500, 698.0783691406250000, 762.1919555664062500,
        826.3055419921875000, 890.4191894531250000, 954.5327758789062500,
        1018.6463623046875000, 1082.7600097656250000, 1146.8735351562500000,
        1210.9871826171875000, 1275.1008300781250000, 1339.2143554687500000,
        1403.3280029296875000, 1467.4415283203125000, 1531.5551757812500000,
        1595.6688232421875000, 1659.7823486328125000, 1723.8959960937500000,
        1788.0096435546875000, 1852.1231689453125000, 1916.2368164062500000,
        1980.3504638671875000, 2044.4639892578125000, 2108.5776367187500000,
        2172.6911621093750000, 2236.8046875000000000, 2300.9182128906250000,
        2365.0319824218750000, 2429.1455078125000000, 2493.2590332031250000,
        2557.3728027343750000, 2621.4863281250000000, 2685.5998535156250000,
        2749.7133789062500000, 2813.8271484375000000
    ]
    self.assertAllEqual(center_freq_exp, center_freq)

    center_freq = filter_bank_ops._calc_center_freq(33, 125, 3800)
    center_freq_exp = [
        243.1058502197265625, 301.0421752929687500, 358.9784851074218750,
        416.9147949218750000, 474.8511352539062500, 532.7874145507812500,
        590.7237548828125000, 648.6600341796875000, 706.5963745117187500,
        764.5327148437500000, 822.4689941406250000, 880.4053344726562500,
        938.3416137695312500, 996.2779541015625000, 1054.2142333984375000,
        1112.1505126953125000, 1170.0869140625000000, 1228.0231933593750000,
        1285.9594726562500000, 1343.8958740234375000, 1401.8321533203125000,
        1459.7684326171875000, 1517.7047119140625000, 1575.6411132812500000,
        1633.5773925781250000, 1691.5136718750000000, 1749.4500732421875000,
        1807.3863525390625000, 1865.3226318359375000, 1923.2589111328125000,
        1981.1953125000000000, 2039.1315917968750000, 2097.0678710937500000
    ]
    self.assertAllEqual(center_freq_exp, center_freq)

    center_freq = filter_bank_ops._calc_center_freq(41, 100, 7500)
    center_freq_exp = [
        214.4616394042968750, 278.4334106445312500, 342.4051513671875000,
        406.3768920898437500, 470.3486328125000000, 534.3204345703125000,
        598.2921752929687500, 662.2639160156250000, 726.2356567382812500,
        790.2073974609375000, 854.1791992187500000, 918.1509399414062500,
        982.1226806640625000, 1046.0944824218750000, 1110.0662841796875000,
        1174.0379638671875000, 1238.0097656250000000, 1301.9814453125000000,
        1365.9532470703125000, 1429.9249267578125000, 1493.8967285156250000,
        1557.8685302734375000, 1621.8402099609375000, 1685.8120117187500000,
        1749.7838134765625000, 1813.7554931640625000, 1877.7272949218750000,
        1941.6990966796875000, 2005.6707763671875000, 2069.6425781250000000,
        2133.6142578125000000, 2197.5861816406250000, 2261.5578613281250000,
        2325.5297851562500000, 2389.5014648437500000, 2453.4731445312500000,
        2517.4450683593750000, 2581.4167480468750000, 2645.3884277343750000,
        2709.3601074218750000, 2773.3320312500000000
    ]
    self.assertAllLess(abs(center_freq_exp - center_freq), 0.00025)

  def testFilterBankStartEndIndices(self):
    start_index, end_index = filter_bank_ops.calc_start_end_indices(
        512, 16000, 32, 125, 3800)
    self.assertEqual(start_index, 5)
    self.assertEqual(end_index, 122)

    start_index, end_index = filter_bank_ops.calc_start_end_indices(
        2048, 44000, 25, 125, 3800)
    self.assertEqual(start_index, 7)
    self.assertEqual(end_index, 177)

    start_index, end_index = filter_bank_ops.calc_start_end_indices(
        512, 16000, 40, 100, 7500)
    self.assertEqual(start_index, 4)
    self.assertEqual(end_index, 241)

  def testFilterBankInitWeight(self):
    (start_index, end_index, weights, unweights, channel_frequency_starts,
     channel_weight_starts,
     channel_widths) = filter_bank_ops._init_filter_bank_weights(
         257, 16000, 1, 1, 32, 125, 3800)
    weights_exp = [
        1133, 2373, 3712, 1047, 2564, 66, 1740, 3486, 1202, 3079, 919, 2913,
        865, 2964, 1015, 3210, 1352, 3633, 1859, 123, 2520, 856, 3323, 1726,
        161, 2722, 1215, 3833, 2382, 956, 3652, 2276, 923, 3689, 2380, 1093,
        3922, 2676, 1448, 239, 3144, 1970, 814, 3770, 2646, 1538, 445, 3463,
        2399, 1349, 313, 3386, 2376, 1379, 394, 3517, 2556, 1607, 668, 3837,
        2920, 2013, 1117, 231, 3450, 2583, 1725, 877, 37, 3302, 2480, 1666,
        861, 63, 3369, 2588, 1813, 1046, 287, 3630, 2885, 2147, 1415, 690,
        4067, 3355, 2650, 1950, 1257, 569, 3984, 3308, 2638, 1973, 1314, 661,
        12, 3465, 2827, 2194, 1566, 944, 325, 3808, 3199, 2595, 1996, 1401,
        810, 224, 3738, 3160, 2586, 2017, 1451, 890, 332
    ]
    unweights_exp = [
        2962,
        1722,
        383,
        3048,
        1531,
        4029,
        2355,
        609,
        2893,
        1016,
        3176,
        1182,
        3230,
        1131,
        3080,
        885,
        2743,
        462,
        2236,
        3972,
        1575,
        3239,
        772,
        2369,
        3934,
        1373,
        2880,
        262,
        1713,
        3139,
        443,
        1819,
        3172,
        406,
        1715,
        3002,
        173,
        1419,
        2647,
        3856,
        951,
        2125,
        3281,
        325,
        1449,
        2557,
        3650,
        632,
        1696,
        2746,
        3782,
        709,
        1719,
        2716,
        3701,
        578,
        1539,
        2488,
        3427,
        258,
        1175,
        2082,
        2978,
        3864,
        645,
        1512,
        2370,
        3218,
        4058,
        793,
        1615,
        2429,
        3234,
        4032,
        726,
        1507,
        2282,
        3049,
        3808,
        465,
        1210,
        1948,
        2680,
        3405,
        28,
        740,
        1445,
        2145,
        2838,
        3526,
        111,
        787,
        1457,
        2122,
        2781,
        3434,
        4083,
        630,
        1268,
        1901,
        2529,
        3151,
        3770,
        287,
        896,
        1500,
        2099,
        2694,
        3285,
        3871,
        357,
        935,
        1509,
        2078,
        2644,
        3205,
        3763,
    ]
    channel_frequency_starts_exp = [
        5, 6, 7, 9, 11, 12, 14, 16, 18, 20, 22, 25, 27, 30, 32, 35, 38, 41, 45,
        48, 52, 56, 60, 64, 69, 74, 79, 84, 89, 95, 102, 108, 115
    ]
    channel_weight_starts_exp = [
        0, 1, 2, 4, 6, 7, 9, 11, 13, 15, 17, 20, 22, 25, 27, 30, 33, 36, 40,
        43, 47, 51, 55, 59, 64, 69, 74, 79, 84, 90, 97, 103, 110
    ]
    channel_widths_exp = [
        1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 4, 3, 4, 4, 4, 4, 5,
        5, 5, 5, 5, 6, 7, 6, 7, 7
    ]
    self.assertEqual(start_index, 5)
    self.assertEqual(end_index, 122)
    self.assertEqual(weights.size, 117)
    self.assertAllEqual(weights, weights_exp)
    self.assertAllEqual(unweights, unweights_exp)
    self.assertAllEqual(channel_frequency_starts, channel_frequency_starts_exp)
    self.assertAllEqual(channel_weight_starts, channel_weight_starts_exp)
    self.assertAllEqual(channel_widths, channel_widths_exp)
    ##################################
    (start_index, end_index, weights, unweights, channel_frequency_starts,
     channel_weight_starts,
     channel_widths) = filter_bank_ops._init_filter_bank_weights(
         257, 16000, 1, 1, 40, 125, 7800)
    weights_exp = [
        1419, 2934, 442, 2130, 3896, 1638, 3546, 1422, 3454, 1449, 3593, 1693,
        3938, 2134, 373, 2751, 1073, 3528, 1925, 356, 2917, 1414, 4037, 2594,
        1180, 3888, 2527, 1191, 3976, 2688, 1422, 179, 3052, 1850, 668, 3601,
        2456, 1329, 220, 3223, 2147, 1087, 42, 3108, 2092, 1091, 103, 3225,
        2263, 1315, 378, 3550, 2638, 1736, 846, 4063, 3195, 2337, 1489, 650,
        3918, 3099, 2289, 1488, 696, 4008, 3233, 2466, 1708, 957, 214, 3575,
        2846, 2126, 1412, 706, 6, 3409, 2723, 2043, 1369, 702, 41, 3482, 2832,
        2189, 1551, 919, 292, 3767, 3151, 2540, 1935, 1334, 739, 148, 3658,
        3077, 2501, 1929, 1362, 799, 240, 3782, 3232, 2686, 2144, 1606, 1073,
        543, 17, 3591, 3072, 2558, 2047, 1539, 1035, 535, 38, 3641, 3151, 2664,
        2180, 1700, 1223, 749, 278, 3906, 3441, 2979, 2520, 2064, 1611, 1161,
        714, 269, 3923, 3484, 3047, 2613, 2182, 1753, 1326, 903, 481, 62, 3742,
        3328, 2916, 2507, 2100, 1695, 1293, 893, 495, 99, 3801, 3410, 3020,
        2633, 2248, 1864, 1483, 1104, 727, 352, 4075, 3703, 3334, 2966, 2601,
        2237, 1875, 1515, 1156, 800, 445, 92, 3836, 3487, 3139, 2792, 2448,
        2105, 1763, 1423, 1085, 749, 414, 80, 3844, 3514, 3185, 2857, 2531,
        2207, 1884, 1562, 1242, 923, 606, 290, 4072, 3758, 3447, 3136, 2827,
        2519, 2213, 1907, 1604, 1301, 999, 699, 400, 103, 3902, 3607, 3313,
        3020, 2729, 2438, 2149, 1861, 1574, 1288, 1003, 720, 437, 156, 3972,
        3693, 3415, 3137, 2862, 2587, 2313, 2040, 1768, 1497, 1228, 959, 691,
        424, 158
    ]
    unweights_exp = [
        2676, 1161, 3653, 1965, 199, 2457, 549, 2673, 641, 2646, 502, 2402,
        157, 1961, 3722, 1344, 3022, 567, 2170, 3739, 1178, 2681, 58, 1501,
        2915, 207, 1568, 2904, 119, 1407, 2673, 3916, 1043, 2245, 3427, 494,
        1639, 2766, 3875, 872, 1948, 3008, 4053, 987, 2003, 3004, 3992, 870,
        1832, 2780, 3717, 545, 1457, 2359, 3249, 32, 900, 1758, 2606, 3445,
        177, 996, 1806, 2607, 3399, 87, 862, 1629, 2387, 3138, 3881, 520, 1249,
        1969, 2683, 3389, 4089, 686, 1372, 2052, 2726, 3393, 4054, 613, 1263,
        1906, 2544, 3176, 3803, 328, 944, 1555, 2160, 2761, 3356, 3947, 437,
        1018, 1594, 2166, 2733, 3296, 3855, 313, 863, 1409, 1951, 2489, 3022,
        3552, 4078, 504, 1023, 1537, 2048, 2556, 3060, 3560, 4057, 454, 944,
        1431, 1915, 2395, 2872, 3346, 3817, 189, 654, 1116, 1575, 2031, 2484,
        2934, 3381, 3826, 172, 611, 1048, 1482, 1913, 2342, 2769, 3192, 3614,
        4033, 353, 767, 1179, 1588, 1995, 2400, 2802, 3202, 3600, 3996, 294,
        685, 1075, 1462, 1847, 2231, 2612, 2991, 3368, 3743, 20, 392, 761,
        1129, 1494, 1858, 2220, 2580, 2939, 3295, 3650, 4003, 259, 608, 956,
        1303, 1647, 1990, 2332, 2672, 3010, 3346, 3681, 4015, 251, 581, 910,
        1238, 1564, 1888, 2211, 2533, 2853, 3172, 3489, 3805, 23, 337, 648,
        959, 1268, 1576, 1882, 2188, 2491, 2794, 3096, 3396, 3695, 3992, 193,
        488, 782, 1075, 1366, 1657, 1946, 2234, 2521, 2807, 3092, 3375, 3658,
        3939, 123, 402, 680, 958, 1233, 1508, 1782, 2055, 2327, 2598, 2867,
        3136, 3404, 3671, 3937
    ]
    channel_frequency_starts_exp = [
        5, 6, 8, 9, 11, 13, 15, 17, 20, 22, 25, 27, 30, 33, 37, 40, 44, 48, 52,
        56, 60, 65, 70, 76, 82, 88, 94, 101, 108, 116, 124, 132, 141, 151, 161,
        171, 183, 195, 207, 221, 235
    ]
    channel_weight_starts_exp = [
        0, 1, 3, 4, 6, 8, 10, 12, 15, 17, 20, 22, 25, 28, 32, 35, 39, 43, 47,
        51, 55, 60, 65, 71, 77, 83, 89, 96, 103, 111, 119, 127, 136, 146, 156,
        166, 178, 190, 202, 216, 230
    ]
    channel_widths_exp = [
        1, 2, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 4, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6,
        6, 6, 7, 7, 8, 8, 8, 9, 10, 10, 10, 12, 12, 12, 14, 14, 15
    ]
    self.assertEqual(start_index, 5)
    self.assertEqual(end_index, 250)
    self.assertEqual(weights.size, 245)
    self.assertAllEqual(weights, weights_exp)
    self.assertAllEqual(unweights, unweights_exp)
    self.assertAllEqual(channel_frequency_starts, channel_frequency_starts_exp)
    self.assertAllEqual(channel_weight_starts, channel_weight_starts_exp)
    self.assertAllEqual(channel_widths, channel_widths_exp)
    ##################################
    (start_index, end_index, weights, unweights, channel_frequency_starts,
     channel_weight_starts,
     channel_widths) = filter_bank_ops._init_filter_bank_weights(
         129, 8000, 1, 1, 25, 125, 3800)
    weights_exp = [
        1762, 3607, 1435, 3431, 1399, 3527, 1619, 3863, 2064, 316, 2710, 1054,
        3536, 1963, 428, 3025, 1562, 132, 2830, 1462, 123, 2909, 1625, 367,
        3230, 2020, 833, 3765, 2621, 1498, 395, 3407, 2341, 1293, 262, 3344,
        2346, 1363, 396, 3539, 2601, 1676, 765, 3963, 3078, 2205, 1343, 494,
        3752, 2925, 2108, 1303, 507, 3817, 3041, 2275, 1517, 769, 30, 3395,
        2673, 1958, 1252, 554, 3959, 3276, 2601, 1932, 1270, 616, 4064, 3423,
        2788, 2160, 1538, 921, 311, 3803, 3205, 2612, 2025, 1443, 866, 295,
        3825, 3264, 2708, 2157, 1611, 1069, 532, 0, 3568, 3044, 2525, 2010,
        1499, 992, 490, 4087, 3592, 3102, 2615, 2131, 1652, 1176, 703, 235,
        3865, 3403, 2945, 2490, 2038, 1589, 1143, 701, 262
    ]
    unweights_exp = [
        2333, 488, 2660, 664, 2696, 568, 2476, 232, 2031, 3779, 1385, 3041,
        559, 2132, 3667, 1070, 2533, 3963, 1265, 2633, 3972, 1186, 2470, 3728,
        865, 2075, 3262, 330, 1474, 2597, 3700, 688, 1754, 2802, 3833, 751,
        1749, 2732, 3699, 556, 1494, 2419, 3330, 132, 1017, 1890, 2752, 3601,
        343, 1170, 1987, 2792, 3588, 278, 1054, 1820, 2578, 3326, 4065, 700,
        1422, 2137, 2843, 3541, 136, 819, 1494, 2163, 2825, 3479, 31, 672,
        1307, 1935, 2557, 3174, 3784, 292, 890, 1483, 2070, 2652, 3229, 3800,
        270, 831, 1387, 1938, 2484, 3026, 3563, 4095, 527, 1051, 1570, 2085,
        2596, 3103, 3605, 8, 503, 993, 1480, 1964, 2443, 2919, 3392, 3860, 230,
        692, 1150, 1605, 2057, 2506, 2952, 3394, 3833
    ]
    channel_frequency_starts_exp = [
        5, 6, 8, 10, 12, 15, 17, 20, 23, 26, 29, 32, 36, 40, 44, 48, 53, 58,
        64, 69, 75, 82, 89, 97, 104, 113
    ]
    channel_weight_starts_exp = [
        0, 1, 3, 5, 7, 10, 12, 15, 18, 21, 24, 27, 31, 35, 39, 43, 48, 53, 59,
        64, 70, 77, 84, 92, 99, 108
    ]
    channel_widths_exp = [
        1, 2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 5, 6, 7, 7, 8, 7,
        9, 9
    ]
    self.assertEqual(start_index, 5)
    self.assertEqual(end_index, 122)
    self.assertEqual(weights.size, 117)
    self.assertAllEqual(weights, weights_exp)
    self.assertAllEqual(unweights, unweights_exp)
    self.assertAllEqual(channel_frequency_starts, channel_frequency_starts_exp)
    self.assertAllEqual(channel_weight_starts, channel_weight_starts_exp)
    self.assertAllEqual(channel_widths, channel_widths_exp)
    ##################################
    (start_index, end_index, weights, unweights, channel_frequency_starts,
     channel_weight_starts,
     channel_widths) = filter_bank_ops._init_filter_bank_weights(
         257, 16000, 2, 1, 25, 125, 3800)
    weights_exp = [
        1762, 3607, 1435, 3431, 1399, 3527, 1619, 3863, 2064, 316, 2710, 1054,
        3536, 1963, 428, 3025, 1562, 132, 2830, 1462, 123, 2909, 1625, 367,
        3230, 2020, 833, 3765, 2621, 1498, 395, 3407, 2341, 1293, 262, 3344,
        2346, 1363, 396, 3539, 2601, 1676, 765, 3963, 3078, 2205, 1343, 494,
        3752, 2925, 2108, 1303, 507, 3817, 3041, 2275, 1517, 769, 30, 3395,
        2673, 1958, 1252, 554, 3959, 3276, 2601, 1932, 1270, 616, 4064, 3423,
        2788, 2160, 1538, 921, 311, 3803, 3205, 2612, 2025, 1443, 866, 295,
        3825, 3264, 2708, 2157, 1611, 1069, 532, 0, 3568, 3044, 2525, 2010,
        1499, 992, 490, 4087, 3592, 3102, 2615, 2131, 1652, 1176, 703, 235,
        3865, 3403, 2945, 2490, 2038, 1589, 1143, 701, 262
    ]
    unweights_exp = [
        2333, 488, 2660, 664, 2696, 568, 2476, 232, 2031, 3779, 1385, 3041,
        559, 2132, 3667, 1070, 2533, 3963, 1265, 2633, 3972, 1186, 2470, 3728,
        865, 2075, 3262, 330, 1474, 2597, 3700, 688, 1754, 2802, 3833, 751,
        1749, 2732, 3699, 556, 1494, 2419, 3330, 132, 1017, 1890, 2752, 3601,
        343, 1170, 1987, 2792, 3588, 278, 1054, 1820, 2578, 3326, 4065, 700,
        1422, 2137, 2843, 3541, 136, 819, 1494, 2163, 2825, 3479, 31, 672,
        1307, 1935, 2557, 3174, 3784, 292, 890, 1483, 2070, 2652, 3229, 3800,
        270, 831, 1387, 1938, 2484, 3026, 3563, 4095, 527, 1051, 1570, 2085,
        2596, 3103, 3605, 8, 503, 993, 1480, 1964, 2443, 2919, 3392, 3860, 230,
        692, 1150, 1605, 2057, 2506, 2952, 3394, 3833
    ]
    channel_frequency_starts_exp = [
        5, 6, 8, 10, 12, 15, 17, 20, 23, 26, 29, 32, 36, 40, 44, 48, 53, 58,
        64, 69, 75, 82, 89, 97, 104, 113
    ]
    channel_weight_starts_exp = [
        0, 1, 3, 5, 7, 10, 12, 15, 18, 21, 24, 27, 31, 35, 39, 43, 48, 53, 59,
        64, 70, 77, 84, 92, 99, 108
    ]
    channel_widths_exp = [
        1, 2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 5, 6, 7, 7, 8, 7,
        9, 9
    ]
    self.assertEqual(start_index, 5)
    self.assertEqual(end_index, 122)
    self.assertEqual(weights.size, 117)
    self.assertAllEqual(weights, weights_exp)
    self.assertAllEqual(unweights, unweights_exp)
    self.assertAllEqual(channel_frequency_starts, channel_frequency_starts_exp)
    self.assertAllEqual(channel_weight_starts, channel_weight_starts_exp)
    self.assertAllEqual(channel_widths, channel_widths_exp)
    ##################################
    (start_index, end_index, weights, unweights, channel_frequency_starts,
     channel_weight_starts,
     channel_widths) = filter_bank_ops._init_filter_bank_weights(
         257, 16000, 1, 1, 40, 100, 7500)
    weights_exp = [
        1875,
        3288,
        702,
        2300,
        3983,
        1647,
        3481,
        1288,
        3255,
        1187,
        3273,
        1317,
        3509,
        1654,
        3941,
        2177,
        455,
        2869,
        1225,
        3714,
        2142,
        603,
        3192,
        1717,
        271,
        2949,
        1558,
        194,
        2951,
        1637,
        346,
        3174,
        1928,
        702,
        3594,
        2409,
        1243,
        96,
        3063,
        1951,
        856,
        3873,
        2810,
        1763,
        731,
        3809,
        2805,
        1815,
        839,
        3971,
        3021,
        2082,
        1156,
        241,
        3434,
        2542,
        1661,
        791,
        4027,
        3177,
        2337,
        1506,
        685,
        3970,
        3167,
        2373,
        1588,
        811,
        43,
        3378,
        2626,
        1881,
        1144,
        414,
        3788,
        3073,
        2365,
        1663,
        969,
        281,
        3696,
        3021,
        2352,
        1689,
        1033,
        382,
        3833,
        3194,
        2560,
        1932,
        1310,
        692,
        80,
        3570,
        2968,
        2371,
        1779,
        1192,
        609,
        32,
        3554,
        2986,
        2422,
        1862,
        1306,
        755,
        208,
        3761,
        3222,
        2687,
        2156,
        1629,
        1106,
        586,
        70,
        3654,
        3146,
        2641,
        2139,
        1641,
        1147,
        656,
        168,
        3779,
        3298,
        2819,
        2344,
        1872,
        1404,
        938,
        475,
        15,
        3654,
        3200,
        2749,
        2300,
        1854,
        1411,
        971,
        534,
        99,
        3762,
        3332,
        2905,
        2480,
        2058,
        1638,
        1221,
        806,
        394,
        4079,
        3671,
        3266,
        2862,
        2461,
        2062,
        1666,
        1271,
        879,
        489,
        100,
        3810,
        3426,
        3044,
        2664,
        2286,
        1910,
        1536,
        1164,
        794,
        425,
        59,
        3790,
        3428,
        3067,
        2707,
        2350,
        1994,
        1640,
        1288,
        938,
        589,
        242,
        3992,
        3649,
        3307,
        2966,
        2627,
        2290,
        1954,
        1620,
        1287,
        956,
        626,
        298,
        4067,
        3742,
        3418,
        3096,
        2775,
        2456,
        2138,
        1821,
        1506,
        1192,
        880,
        568,
        259,
        4046,
        3739,
        3433,
        3128,
        2825,
        2523,
        2222,
        1923,
        1624,
        1327,
        1031,
        737,
        443,
        151,
        3956,
        3666,
        3377,
        3090,
        2803,
        2518,
        2234,
        1951,
        1669,
        1388,
        1108,
        829,
        552,
        275,
        0,
    ]
    unweights_exp = [
        2220,
        807,
        3393,
        1795,
        112,
        2448,
        614,
        2807,
        840,
        2908,
        822,
        2778,
        586,
        2441,
        154,
        1918,
        3640,
        1226,
        2870,
        381,
        1953,
        3492,
        903,
        2378,
        3824,
        1146,
        2537,
        3901,
        1144,
        2458,
        3749,
        921,
        2167,
        3393,
        501,
        1686,
        2852,
        3999,
        1032,
        2144,
        3239,
        222,
        1285,
        2332,
        3364,
        286,
        1290,
        2280,
        3256,
        124,
        1074,
        2013,
        2939,
        3854,
        661,
        1553,
        2434,
        3304,
        68,
        918,
        1758,
        2589,
        3410,
        125,
        928,
        1722,
        2507,
        3284,
        4052,
        717,
        1469,
        2214,
        2951,
        3681,
        307,
        1022,
        1730,
        2432,
        3126,
        3814,
        399,
        1074,
        1743,
        2406,
        3062,
        3713,
        262,
        901,
        1535,
        2163,
        2785,
        3403,
        4015,
        525,
        1127,
        1724,
        2316,
        2903,
        3486,
        4063,
        541,
        1109,
        1673,
        2233,
        2789,
        3340,
        3887,
        334,
        873,
        1408,
        1939,
        2466,
        2989,
        3509,
        4025,
        441,
        949,
        1454,
        1956,
        2454,
        2948,
        3439,
        3927,
        316,
        797,
        1276,
        1751,
        2223,
        2691,
        3157,
        3620,
        4080,
        441,
        895,
        1346,
        1795,
        2241,
        2684,
        3124,
        3561,
        3996,
        333,
        763,
        1190,
        1615,
        2037,
        2457,
        2874,
        3289,
        3701,
        16,
        424,
        829,
        1233,
        1634,
        2033,
        2429,
        2824,
        3216,
        3606,
        3995,
        285,
        669,
        1051,
        1431,
        1809,
        2185,
        2559,
        2931,
        3301,
        3670,
        4036,
        305,
        667,
        1028,
        1388,
        1745,
        2101,
        2455,
        2807,
        3157,
        3506,
        3853,
        103,
        446,
        788,
        1129,
        1468,
        1805,
        2141,
        2475,
        2808,
        3139,
        3469,
        3797,
        28,
        353,
        677,
        999,
        1320,
        1639,
        1957,
        2274,
        2589,
        2903,
        3215,
        3527,
        3836,
        49,
        356,
        662,
        967,
        1270,
        1572,
        1873,
        2172,
        2471,
        2768,
        3064,
        3358,
        3652,
        3944,
        139,
        429,
        718,
        1005,
        1292,
        1577,
        1861,
        2144,
        2426,
        2707,
        2987,
        3266,
        3543,
        3820,
        4096,
    ]
    channel_frequency_starts_exp = [
        4, 5, 7, 8, 10, 12, 14, 16, 18, 21, 23, 26, 29, 32, 35, 38, 42, 45, 49,
        53, 58, 62, 67, 73, 78, 84, 90, 97, 104, 111, 119, 127, 136, 145, 154,
        165, 176, 187, 199, 212, 226
    ]
    channel_weight_starts_exp = [
        0,
        1,
        3,
        4,
        6,
        8,
        10,
        12,
        14,
        17,
        19,
        22,
        25,
        28,
        31,
        34,
        38,
        41,
        45,
        49,
        54,
        58,
        63,
        69,
        74,
        80,
        86,
        93,
        100,
        107,
        115,
        123,
        132,
        141,
        150,
        161,
        172,
        183,
        195,
        208,
        222,
    ]
    channel_widths_exp = [
        1,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        3,
        4,
        4,
        5,
        4,
        5,
        6,
        5,
        6,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        9,
        9,
        11,
        11,
        11,
        12,
        13,
        14,
        15,
    ]
    self.assertEqual(start_index, 4)
    self.assertEqual(end_index, 241)
    self.assertEqual(weights.size, 237)
    self.assertAllLessEqual(abs(weights - weights_exp), 1)
    self.assertAllLessEqual(abs(unweights - unweights_exp), 1)
    self.assertAllEqual(channel_frequency_starts, channel_frequency_starts_exp)
    self.assertAllEqual(channel_weight_starts, channel_weight_starts_exp)
    self.assertAllEqual(channel_widths, channel_widths_exp)

  def SingleFilterBankSpectralSubtractionVectorTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    num_channels = int(args[0])
    smoothing = float(args[1])
    alternate_smoothing = float(args[2])
    smoothing_bits = int(args[3])
    min_signal_remaining = float(args[4])
    clamping = bool(int(args[5]))

    func = tf.function(filter_bank_ops.filter_bank_spectral_subtraction)
    input_size = len(lines[1].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.uint32), num_channels, smoothing,
        alternate_smoothing, smoothing_bits, min_signal_remaining, clamping)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in triplets <input, expected output, expected noise estimate>
    i = 1
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype=np.uint32)
      out_frame_exp = [int(j) for j in lines[i + 1].split()]
      noise_estimate_exp = [int(j) for j in lines[i + 2].split()]
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      noise_estimate = interpreter.get_output(1)
      self.assertAllEqual(out_frame, out_frame_exp)
      self.assertAllEqual(noise_estimate, noise_estimate_exp)
      # TF
      [out_frame, noise_estimate] = self.evaluate(
          filter_bank_ops.filter_bank_spectral_subtraction(
              in_frame, num_channels, smoothing, alternate_smoothing,
              smoothing_bits, min_signal_remaining, clamping))
      self.assertAllEqual(out_frame, out_frame_exp)
      self.assertAllEqual(noise_estimate, noise_estimate_exp)
      i += 3

  def SingleFilterBankSquareRootVectorTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    func = tf.function(filter_bank_ops.filter_bank_square_root)
    input_size = len(lines[0].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.uint64),
        tf.TensorSpec([], dtype=tf.int32))
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Read lines in triplets <input, scale bits, expected output>
    i = 0
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype=np.uint64)
      scale_bits = np.array(int(lines[i + 1]), dtype=np.int32)
      out_frame_exp = [int(j) for j in lines[i + 2].split()]
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.set_input(scale_bits, 1)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      self.assertAllEqual(out_frame, out_frame_exp)
      # TF
      out_frame = self.evaluate(
          filter_bank_ops.filter_bank_square_root(in_frame, scale_bits))
      self.assertAllEqual(out_frame, out_frame_exp)
      i += 3

  def SingleFilterBankVectorTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    sample_rate = int(args[0])
    num_channels = int(args[1])
    lower_band_limit = float(args[2])
    upper_band_limit = float(args[3])
    func = tf.function(filter_bank_ops.filter_bank)
    input_size = len(lines[1].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.uint32), sample_rate, num_channels,
        lower_band_limit, upper_band_limit)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in pairs <input, expected output>
    i = 1
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype=np.uint32)
      out_frame_exp = [int(j) for j in lines[i + 1].split()]
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      self.assertAllEqual(out_frame_exp, out_frame)
      # TF
      out_frame = self.evaluate(
          filter_bank_ops.filter_bank(in_frame, sample_rate, num_channels,
                                      lower_band_limit, upper_band_limit))
      self.assertAllEqual(out_frame_exp, out_frame)
      i += 2

  def SingleFilterBankTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    input_tensor = np.arange(int(args[0]), dtype=np.uint32)
    output_exp = [int(i) for i in lines[1:]]
    output_tensor = self.evaluate(
        filter_bank_ops.filter_bank(input_tensor, int(args[1]), int(args[4]),
                                    float(args[2]), float(args[3])))
    self.assertAllLessEqual(abs(output_exp - output_tensor), 144)

  def testFilterBank(self):
    self.SingleFilterBankTest('testdata/filter_bank_accumulation_8k.txt')
    self.SingleFilterBankTest('testdata/filter_bank_accumulation_16k.txt')
    self.SingleFilterBankTest('testdata/filter_bank_accumulation_44k.txt')
    self.SingleFilterBankVectorTest('testdata/filter_bank_test1.txt')

  def testFilterBankSpectralSubtractionVector(self):
    self.SingleFilterBankSpectralSubtractionVectorTest(
        'testdata/filter_bank_spectral_subtraction_test1.txt')

  def testFilterBankSquareRootVector(self):
    self.SingleFilterBankSquareRootVectorTest(
        'testdata/filter_bank_square_root_test1.txt')

  def testFilterBankSquareRoot(self):
    fft_scale_bits = 7
    input_array = [
        632803382, 3322331443, 7096652410, 7915374281, 1173754459, 305980674,
        2000536077, 1168558488, 5076475823, 15976754090, 3805664731, 613998164,
        1697378269, 2775934843, 3579468406, 2317762617, 2025182819, 3166301049,
        1937595023, 1774351019, 2085308695, 3187965791, 2871034131, 4396421345,
        8203017514, 4506083115, 3159809690, 750384531, 243621165, 61552427,
        794881, 285365, 324568, 209218, 212215, 311565, 183541, 223754, 201098,
        385031
    ]
    output_exp = [
        196, 450, 658, 695, 267, 136, 349, 267, 556, 987, 481, 193, 321, 411,
        467, 376, 351, 439, 343, 329, 356, 441, 418, 518, 707, 524, 439, 214,
        121, 61, 6, 4, 4, 3, 3, 4, 3, 3, 3, 4
    ]
    output_array = self.evaluate(
        filter_bank_ops.filter_bank_square_root(input_array, fft_scale_bits))
    self.assertAllEqual(output_array, output_exp)

    fft_scale_bits = 2
    input_array = [
        1384809583, 3253852150, 7271882261, 4247132793, 165951197, 106924444,
        334793989, 1186792065, 683710887, 328783218, 1777824058, 859450346,
        384515125, 118491239, 29264336, 324188526, 1925807083, 2591551091,
        1170412774, 393317159, 1003847215, 1375415668, 1272433002, 5102945913,
        5527301760, 3564304855, 4171837220, 4252817101, 2886468276, 1293586339,
        867722874, 137636997
    ]
    output_exp = [
        9303, 14260, 21318, 16292, 3220, 2585, 4574, 8612, 6537, 4533, 10541,
        7329, 4902, 2721, 1352, 4501, 10971, 12726, 8552, 4958, 7921, 9271,
        8917, 17858, 18586, 14925, 16147, 16303, 13431, 8991, 7364, 2933
    ]
    output_array = self.evaluate(
        filter_bank_ops.filter_bank_square_root(input_array, fft_scale_bits))
    self.assertAllEqual(output_array, output_exp)

  def testFilterBankLog(self):
    output_scale = 1600
    correction_bits = 3
    input_array = [
        29, 21, 29, 40, 19, 11, 13, 23, 13, 11, 25, 17, 5, 4, 46, 14, 17, 14,
        20, 14, 10, 10, 15, 11, 17, 12, 15, 16, 19, 18, 6, 2
    ]
    output_exp = [
        8715, 8198, 8715, 9229, 8038, 7164, 7431, 8344, 7431, 7164, 8477, 7860,
        5902, 5545, 9453, 7550, 7860, 7550, 8120, 7550, 7011, 7011, 7660, 7164,
        7860, 7303, 7660, 7763, 8038, 7952, 6194, 4436
    ]
    output_array = self.evaluate(
        filter_bank_ops.filter_bank_log(input_array, output_scale,
                                        correction_bits))
    self.assertAllEqual(output_array, output_exp)


if __name__ == '__main__':
  tf.test.main()
