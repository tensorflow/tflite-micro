// Example code illustrating the theory exposed in doc/quantization.md

/* Command line to build and run on x86:

c++ doc/quantization_example.cc -I . --std=c++11 -msse4.1 -lpthread \
  -o /tmp/quantization_example && \
/tmp/quantization_example

*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "../public/gemmlowp.h"
#include "../public/output_stages.h"

// We will handle both float and quantized matrices, which we will
// represent as gemmlowp::MatrixMap.
// We will need to be able to print them.

// Output a matrix to a std::ostream
template <typename tScalar, gemmlowp::MapOrder tOrder>
std::ostream& operator<<(std::ostream& s,
                         const gemmlowp::MatrixMap<tScalar, tOrder>& m) {
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      if (j) {
        s << '\t';
      }
      s << static_cast<float>(m(i, j));
    }
    s << '\n';
  }
  return s;
}

// Find the min and max value in a float matrix.
template <gemmlowp::MapOrder tOrder>
void FindMinMax(const gemmlowp::MatrixMap<float, tOrder>& m, float* min,
                float* max) {
  *min = *max = m(0, 0);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      const float val = m(i, j);
      *min = std::min(*min, val);
      *max = std::max(*max, val);
    }
  }
}

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct QuantizationParams {
  float scale;
  std::uint8_t zero_point;
};

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max) {
  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // the min and max quantized values, as floating-point values
  const float qmin = 0;
  const float qmax = 255;

  // First determine the scale.
  const double scale = (max - min) / (qmax - qmin);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // Let's use the first one here.
  const double initial_zero_point = qmin - min / scale;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  std::uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point =
        static_cast<std::uint8_t>(std::round(initial_zero_point));
  }

  QuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}

template <gemmlowp::MapOrder tLhsOrder, gemmlowp::MapOrder tRhsOrder,
          gemmlowp::MapOrder tResultOrder>
void FloatMatrixMultiplication(
    const gemmlowp::MatrixMap<const float, tLhsOrder>& lhs,
    const gemmlowp::MatrixMap<const float, tRhsOrder>& rhs,
    gemmlowp::MatrixMap<float, tResultOrder>* result) {
  assert(lhs.cols() == rhs.rows());
  assert(lhs.rows() == result->rows());
  assert(rhs.cols() == result->cols());
  for (int i = 0; i < lhs.rows(); i++) {
    for (int k = 0; k < rhs.cols(); k++) {
      (*result)(i, k) = 0;
      for (int j = 0; j < lhs.cols(); j++) {
        (*result)(i, k) += lhs(i, j) * rhs(j, k);
      }
    }
  }
}

void Quantize(const QuantizationParams& qparams, const std::vector<float>& src,
              std::vector<std::uint8_t>* dst) {
  assert(src.size() == dst->size());
  for (std::size_t i = 0; i < src.size(); i++) {
    const float real_val = src[i];
    const float transformed_val = qparams.zero_point + real_val / qparams.scale;
    const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
    (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));
  }
}

void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::uint8_t>& src, std::vector<float>* dst) {
  assert(src.size() == dst->size());
  for (std::size_t i = 0; i < src.size(); i++) {
    const std::uint8_t quantized_val = src[i];
    (*dst)[i] = qparams.scale * (quantized_val - qparams.zero_point);
  }
}

template <typename tScalar, gemmlowp::MapOrder tOrder>
class MatrixWithStorage {
 public:
  MatrixWithStorage(int rows, int cols)
      : storage(rows * cols), matrix_map(storage.data(), rows, cols) {}
  void MakeRandom() {
    static std::mt19937 random_engine;
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (auto& x : storage) {
      x = static_cast<tScalar>(distribution(random_engine));
    }
  }
  gemmlowp::MatrixMap<const tScalar, tOrder> ConstMap() const {
    return gemmlowp::MatrixMap<const tScalar, tOrder>(
        storage.data(), matrix_map.rows(), matrix_map.cols());
  }
  gemmlowp::MatrixMap<tScalar, tOrder> Map() {
    return gemmlowp::MatrixMap<tScalar, tOrder>(
        storage.data(), matrix_map.rows(), matrix_map.cols());
  }
  const std::vector<tScalar>& Storage() const { return storage; }
  std::vector<tScalar>& Storage() { return storage; }

 private:
  std::vector<tScalar> storage;
  gemmlowp::MatrixMap<tScalar, tOrder> matrix_map;
};

template <typename tScalar, gemmlowp::MapOrder tOrder>
std::ostream& operator<<(std::ostream& s,
                         const MatrixWithStorage<tScalar, tOrder>& m) {
  return s << m.ConstMap();
}

// Given a real_multiplier in the interval (0, 1),
// produces a pair (quantized_multiplier, right_shift) where
// quantized_multiplier is an int32 representing a fixed-point value
// in the interval [-1, 1)  (in practice we only produce positive values)
// and right_shift is an amount to shift right by, so that the
// floating-point multiplication of some int32 input value by real_multiplier,
//
//   return static_cast<int32>(int32_value * real_multiplier);
//
// is best approximated by the integer-arithmetic-only code
//
//   return RoundingRightShift(
//       FixedPointMultiplication(int32_value, quantized_multiplier),
//       right_shift);
//
// This is how to obtain the fixed-point multiplier and right shift
// parameters to pass to
// OutputStageQuantizeDownInt32ByFixedPoint.
//
// Note: all this code only needs to run offline to generate the quantized
// neural network workload, not at runtime on the
// device on which quantized neural networks need to run. So it's not
// performance-critical at all.
void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                      std::int32_t* quantized_multiplier,
                                      int* right_shift) {
  assert(real_multiplier > 0.f);
  assert(real_multiplier < 1.f);
  int s = 0;
  // We want to bring the real multiplier into the interval [1/2, 1).
  // We can do so by multiplying it by two, and recording how many times
  // we multiplied by two so that we can compensate that by a right
  // shift by the same amount.
  while (real_multiplier < 0.5f) {
    real_multiplier *= 2.0f;
    s++;
  }
  // Now that the real multiplier is in [1/2, 1), we convert it
  // into a fixed-point number.
  std::int64_t q =
      static_cast<std::int64_t>(std::round(real_multiplier * (1ll << 31)));
  assert(q <= (1ll << 31));
  // Handle the special case when the real multiplier was so close to 1
  // that its fixed-point approximation was undistinguishable from 1.
  // We handle this by dividing it by two, and remembering to decrement
  // the right shift amount.
  if (q == (1ll << 31)) {
    q /= 2;
    s--;
  }
  assert(s >= 0);
  assert(q <= std::numeric_limits<std::int32_t>::max());
  *quantized_multiplier = static_cast<std::int32_t>(q);
  *right_shift = s;
}

int main() {
  std::cout.precision(3);

  const int rows = 2;
  const int depth = 4;
  const int cols = 3;
  const auto kOrder = gemmlowp::MapOrder::ColMajor;

  std::cout << "First, let us make some float matrices LHS and RHS, "
            << "and compute their product.\n"
            << std::endl;
  MatrixWithStorage<float, kOrder> float_lhs(rows, depth);
  float_lhs.MakeRandom();
  MatrixWithStorage<float, kOrder> float_rhs(depth, cols);
  float_rhs.MakeRandom();
  MatrixWithStorage<float, kOrder> reference_float_result(rows, cols);
  auto reference_float_result_map = reference_float_result.Map();
  FloatMatrixMultiplication(float_lhs.ConstMap(), float_rhs.ConstMap(),
                            &reference_float_result_map);
  std::cout << "Here is the float LHS matrix:\n" << float_lhs << std::endl;
  std::cout << "Here is the float RHS matrix:\n" << float_rhs << std::endl;
  std::cout << "Here is the float product (LHS * RHS) matrix obtained by "
            << "ordinary float matrix multiplication, i.e. as far as we are "
            << "concerned, the REFERENCE RESULT:\n"
            << reference_float_result << std::endl;

  std::cout
      << "Now we embark on reproducing this result using "
      << "quantized arithmetic. The code below splits into two parts: "
      << "quantization code that only needs to run offline (e.g. to "
      << "generate a quantized neural network workload), and actual "
      << "runtime quantized code, which is typically performance-critical "
      << "and where we typically do not want to use any floating-point "
      << "arithmetic. We want to clearly distinguish between the two.\n"
      << std::endl;

  std::cout << "The below is OFFLINE QUANTIZATION CODE. We still use some "
            << "floating-point arithmetic in the process of generating the "
            << "quantized workload to be run on-device.\n"
            << std::endl;

  std::cout
      << "Now, let us choose quantization parameters for these matrices. "
      << "You might ask, what good is quantization if we need to pick "
      << "quantization parameters for the result before we can run the "
      << "quantized computation to obtain the result? The idea is that we "
      << "target applications such as neural networks, where unknown results "
      << "are only allowed to vary within preexisting bounds. In practice, the "
      << "bounds for the results are typically learned during the neural "
         "network "
      << "training process. The min and max of the result do not have to be "
      << "exact. If they are too broad, we just get lower quantization "
         "accuracy. "
      << "If they are too narrow, we just get clamping at the bounds.\n"
      << std::endl;

  float lhs_min, lhs_max, rhs_min, rhs_max, result_min, result_max;
  FindMinMax(float_lhs.Map(), &lhs_min, &lhs_max);
  FindMinMax(float_rhs.Map(), &rhs_min, &rhs_max);
  FindMinMax(reference_float_result.Map(), &result_min, &result_max);
  const auto lhs_qparams = ChooseQuantizationParams(lhs_min, lhs_max);
  const auto rhs_qparams = ChooseQuantizationParams(rhs_min, rhs_max);
  const auto result_qparams = ChooseQuantizationParams(result_min, result_max);

  std::cout << "For LHS, we have min = " << lhs_min << ", max = " << lhs_max
            << ", scale = " << lhs_qparams.scale
            << ", zero_point = " << static_cast<float>(lhs_qparams.zero_point)
            << std::endl;
  std::cout << "For RHS, we have min = " << rhs_min << ", max = " << rhs_max
            << ", scale = " << rhs_qparams.scale
            << ", zero_point = " << static_cast<float>(rhs_qparams.zero_point)
            << std::endl;
  std::cout << "For the result, we have min = " << result_min
            << ", max = " << result_max << ", scale = " << result_qparams.scale
            << ", zero_point = "
            << static_cast<float>(result_qparams.zero_point) << std::endl;

  std::cout << std::endl;

  MatrixWithStorage<std::uint8_t, kOrder> uint8_lhs(rows, depth);
  MatrixWithStorage<std::uint8_t, kOrder> uint8_rhs(depth, cols);
  MatrixWithStorage<std::uint8_t, kOrder> actual_uint8_result(rows, cols);

  Quantize(lhs_qparams, float_lhs.Storage(), &uint8_lhs.Storage());
  Quantize(rhs_qparams, float_rhs.Storage(), &uint8_rhs.Storage());

  std::cout << "Quantized uint8 LHS matrix:\n" << uint8_lhs << std::endl;
  std::cout << "Quantized uint8 RHS matrix:\n" << uint8_rhs << std::endl;

  const int lhs_offset = -lhs_qparams.zero_point;
  const int rhs_offset = -rhs_qparams.zero_point;
  const int result_offset = result_qparams.zero_point;

  const float real_multiplier =
      lhs_qparams.scale * rhs_qparams.scale / result_qparams.scale;
  std::int32_t quantized_multiplier;
  int right_shift;
  QuantizeMultiplierSmallerThanOne(real_multiplier, &quantized_multiplier,
                                   &right_shift);

  std::cout << "End of OFFLINE QUANTIZATION CODE.\n" << std::endl;

  std::cout << "The below is ON-DEVICE RUNTIME QUANTIZED CODE. "
            << "This is the part that is performance-critical and may only "
            << "use quantized arithmetic.\n"
            << std::endl;

  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint
      quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = result_offset;
  quantize_down_stage.result_fixedpoint_multiplier = quantized_multiplier;
  quantize_down_stage.result_shift = right_shift;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  const auto& output_pipeline =
      std::make_tuple(quantize_down_stage, saturating_cast_stage);

  auto actual_uint8_result_map = actual_uint8_result.Map();
  gemmlowp::GemmContext gemm_context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, uint8_lhs.ConstMap(), uint8_rhs.ConstMap(),
      &actual_uint8_result_map, lhs_offset, rhs_offset, output_pipeline);

  std::cout << "Quantized uint8 result matrix obtained by quantized "
            << "multiplication:\n"
            << actual_uint8_result << std::endl;

  std::cout << "End of ON-DEVICE RUNTIME QUANTIZED CODE.\n" << std::endl;

  MatrixWithStorage<float, kOrder> actual_float_result(rows, cols);
  Dequantize(result_qparams, actual_uint8_result.Storage(),
             &actual_float_result.Storage());
  std::cout
      << "Here is the actual float product (LHS * RHS) matrix obtained by "
      << "dequantizing the above uint8 result, i.e. "
      << "as far as we are concerned, the ACTUAL RESULT:\n"
      << actual_float_result << std::endl;

  MatrixWithStorage<float, kOrder> diff_float_result(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      diff_float_result.Map()(i, j) =
          actual_float_result.Map()(i, j) - reference_float_result.Map()(i, j);
    }
  }

  std::cout << "Difference between ACTUAL and REFERENCE float results:\n"
            << diff_float_result << std::endl;
}