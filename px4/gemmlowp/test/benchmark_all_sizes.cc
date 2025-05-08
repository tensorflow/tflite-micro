// Example command line to build on Android ARM64:
/*
~/android/toolchains/r15c-aarch64/bin/aarch64-linux-android-clang++ \
test/benchmark_all_sizes.cc -o /tmp/b -O3 --std=c++11 -fPIE -static \
-DBENCHMARK_QUICK -DBENCHMARK_8bit
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "../public/gemmlowp.h"

#ifdef GEMMLOWP_PROFILING
#include "../profiling/profiler.h"
#endif

#if defined GEMMLOWP_ANDROID && defined GEMMLOWP_ARM_32
// Compilation workaround
namespace std {
  using ::round;
}
#endif

// Minimum duration of each benchmark measurement. Also, duration
// of sleep time between each two consecutive benchmark measurements to
// prevent over-heating.
const double kBenchmarkSecs = 0.1;

// Sleep time before each benchmark.
const int kCooldownBeforeBenchmarkSecs = 0;

// Number of benchmark passes.
const int kPasses = 4;

#ifdef BENCHMARK_NUM_THREADS
const int kNumThreads = BENCHMARK_NUM_THREADS;
#else
const int kNumThreads = 1;
#endif

namespace gemmlowp {

// gemmlowp itself doesn't have a Matrix class, only a MatrixMap class,
// since it only maps existing data. In tests though, we need to
// create our own matrices.
template <typename tScalar, MapOrder tOrder>
class Matrix : public MatrixMap<tScalar, tOrder> {
 public:
  typedef MatrixMap<tScalar, tOrder> Map;
  typedef MatrixMap<const tScalar, tOrder> ConstMap;
  typedef typename Map::Scalar Scalar;
  static const MapOrder Order = tOrder;
  using Map::cols_;
  using Map::data_;
  using Map::kOrder;
  using Map::rows_;
  using Map::stride_;

 public:
  Matrix() : Map(nullptr, 0, 0, 0) {}

  Matrix(int rows, int cols) : Map(nullptr, 0, 0, 0) { Resize(rows, cols); }

  Matrix(const Matrix& other) : Map(nullptr, 0, 0, 0) { *this = other; }

  Matrix& operator=(const Matrix& other) {
    Resize(other.rows_, other.cols_);
    std::memcpy(data_, other.data_, size() * sizeof(Scalar));
    return *this;
  }

  friend bool operator==(const Matrix& a, const Matrix& b) {
    return a.rows_ == b.rows_ && a.cols_ == b.cols_ &&
           !std::memcmp(a.data_, b.data_, a.size());
  }

  void Resize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    stride_ = kOrder == MapOrder::ColMajor ? rows : cols;
    storage.resize(size());
    data_ = storage.data();
  }

  int size() const { return rows_ * cols_; }

  Map& map() { return *static_cast<Map*>(this); }

  ConstMap const_map() const { return ConstMap(data_, rows_, cols_, stride_); }

 protected:
  std::vector<Scalar> storage;
};

template <typename MatrixType>
void MakeZero(MatrixType* m) {
  for (int c = 0; c < m->cols(); c++) {
    for (int r = 0; r < m->rows(); r++) {
      (*m)(r, c) = 128;
    }
  }
}

}  // end namespace gemmlowp

template <typename BitDepthParams>
float benchmark_8bit(int rows, int depth, int cols) {
  using namespace gemmlowp;
  typedef Matrix<std::uint8_t, MapOrder::RowMajor> LhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> RhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> ResultType;

  LhsType lhs;
  RhsType rhs;
  ResultType result;
  lhs.Resize(rows, depth);
  rhs.Resize(depth, cols);
  result.Resize(rows, cols);
  MakeZero(&lhs);
  MakeZero(&rhs);
  MakeZero(&result);

  typedef std::tuple<OutputStageQuantizeDownInt32ByFixedPoint,
                     OutputStageSaturatingCastToUint8>
      Pipeline;
  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint
      quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = 128;
  quantize_down_stage.result_fixedpoint_multiplier = 1234567890;
  quantize_down_stage.result_shift = 16;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  const auto output_pipeline =
      std::make_tuple(quantize_down_stage, saturating_cast_stage);
  GemmContext gemm_context;
  gemm_context.set_max_num_threads(kNumThreads);
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t, BitDepthParams>(
      &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
      -128, output_pipeline);

  double time_start = real_time_in_seconds();
  double t = time_start;
  int iters = 0;
  int iters_at_a_time = 1;
  while (t - time_start < kBenchmarkSecs) {
    for (int i = 0; i < iters_at_a_time; i++) {
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                       BitDepthParams>(
          &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
          -128, output_pipeline);
      iters++;
    }
    iters_at_a_time *= 2;
    t = real_time_in_seconds();
  }
  return (t - time_start) / iters;
}

template <typename BitDepthParams>
float benchmark_8bit_to_32bit(int rows, int depth, int cols) {
  using namespace gemmlowp;
  typedef Matrix<std::uint8_t, MapOrder::RowMajor> LhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> RhsType;
  typedef Matrix<std::int32_t, MapOrder::ColMajor> ResultType;

  LhsType lhs;
  RhsType rhs;
  ResultType result;
  lhs.Resize(rows, depth);
  rhs.Resize(depth, cols);
  result.Resize(rows, cols);
  MakeZero(&lhs);
  MakeZero(&rhs);
  MakeZero(&result);

  typedef std::tuple<> EmptyPipeline;
  GemmContext gemm_context;
  gemm_context.set_max_num_threads(kNumThreads);
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t, BitDepthParams>(
      &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
      -128, EmptyPipeline());

  double time_start = real_time_in_seconds();
  double t = time_start;
  int iters = 0;
  int iters_at_a_time = 1;
  while (t - time_start < kBenchmarkSecs) {
    for (int i = 0; i < iters_at_a_time; i++) {
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                       BitDepthParams>(
          &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
          -128, EmptyPipeline());
      iters++;
    }
    iters_at_a_time *= 2;
    t = real_time_in_seconds();
  }
  return (t - time_start) / iters;
}

struct Shape {
  int rows;
  int depth;
  int cols;
};

bool operator==(const Shape& s1, const Shape& s2) {
  return s1.rows == s2.rows && s1.depth == s2.depth && s1.cols == s2.cols;
}

bool operator<(const Shape& shape1, const Shape& shape2) {
  return shape1.depth < shape2.depth ||
         (shape1.depth == shape2.depth &&
          (shape1.rows < shape2.rows ||
           (shape1.rows == shape2.rows && shape1.cols < shape2.cols)));
};

#ifdef _WIN32
#define sleep(t) Sleep(t)
#endif

float benchmark(const Shape& shape) {
  if (kCooldownBeforeBenchmarkSecs) {
    sleep(kCooldownBeforeBenchmarkSecs);
  }
#if defined BENCHMARK_8bit
  // Benchmark the fast 8bit path, using L8R8WithLhsNonzeroBitDepthParams.
  // This is the recommended thing to default to: it's what most applications
  // want to use, as it's the fastest.
  // The contract is that LHS must take values in [1, 255], while RHS can take
  // any value in [0, 255].
  return benchmark_8bit<gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      shape.rows, shape.depth, shape.cols);
#elif defined BENCHMARK_8bit_wide
  // Variant benchmarking the slower (mostly legacy) DefaultL8R8BitDepthParams.
  // The only contract difference is that both LHS and RHS can take values in
  // [0, 255].
  return benchmark_8bit<gemmlowp::DefaultL8R8BitDepthParams>(
      shape.rows, shape.depth, shape.cols);
#elif defined BENCHMARK_8bit_to_32bit
  // Variant of BENCHMARK_8bit where the user asks for getting raw int32
  // accumulators, instead of a 8bit-downscaled result.
  return benchmark_8bit_to_32bit<gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      shape.rows, shape.depth, shape.cols);
#elif defined BENCHMARK_8bit_to_32bit_wide
  // Variant of BENCHMARK_8bit_wide where the user asks for getting raw int32
  // accumulators, instead of a 8bit-downscaled result.
  return benchmark_8bit_to_32bit<gemmlowp::DefaultL8R8BitDepthParams>(
      shape.rows, shape.depth, shape.cols);
#elif defined BENCHMARK_float
  return benchmark_float(shape.rows, shape.depth, shape.cols);
#else
#error What arithmetic path should we benchmark? (Suggestion: #define BENCHMARK_8bit)
#endif
}

std::set<int> all_sizes() {
  std::set<int> sizes;
  for (int i = 1; i <= 2048; i *= 2) {
    sizes.insert(i);
  }
  for (double x = 8; x <= 2048; x *= std::sqrt(2.)) {
    sizes.insert(static_cast<int>(std::round(x)));
  }
  for (double x = 16; x <= 512; x *= std::pow(2., 1. / 4.)) {
    sizes.insert(static_cast<int>(std::round(x)));
  }
  return sizes;
}

std::mt19937& RandomEngine() {
  static std::mt19937 engine;
  return engine;
}

std::vector<Shape> all_shapes_in_random_order() {
  std::vector<Shape> shapes;
  const std::set<int> sizes = all_sizes();
#if defined BENCHMARK_ROWS
  // Benchmark one specific shape
  Shape shape;
  shape.rows = BENCHMARK_ROWS;
  shape.depth = BENCHMARK_DEPTH;
  shape.cols = BENCHMARK_COLS;
  shapes.push_back(shape);
#elif defined BENCHMARK_QUICK
  // Benchmark an assortment of cubic shapes
  for (int size : sizes) {
    Shape shape;
    shape.rows = size;
    shape.depth = size;
    shape.cols = size;
    shapes.push_back(shape);
  }
#elif defined BENCHMARK_EXHAUSTIVE
  // Benchmark all sorts of shapes
  for (int rows : sizes) {
    for (int depth : sizes) {
      for (int cols : sizes) {
        Shape shape;
        shape.rows = rows;
        shape.depth = depth;
        shape.cols = cols;
        shapes.push_back(shape);
      }
    }
  }
#else
#error What shapes should we benchmark? (Suggestion: #define BENCHMARK_QUICK)
#endif
  std::shuffle(std::begin(shapes), std::end(shapes), RandomEngine());
  return shapes;
}

void run_benchmarks(std::map<Shape, float>* results) {
  std::vector<Shape> shapes;
  for (int pass = 0; pass < kPasses; pass++) {
    const std::vector<Shape> pass_shapes = all_shapes_in_random_order();
    shapes.insert(std::end(shapes), std::begin(pass_shapes),
                  std::end(pass_shapes));
  }

  const double time_start = gemmlowp::real_time_in_seconds();
  for (std::size_t i = 0; i < shapes.size(); i++) {
    const double ratio = static_cast<double>(i) / shapes.size();
    const double elapsed = gemmlowp::real_time_in_seconds() - time_start;
    const double elapsed_hours = elapsed / 3600.;
    const double eta_hours = elapsed_hours * (1. - ratio) / ratio;
    fprintf(stderr,
            "Benchmarking: %.2f%% done, Elapsed: %.2f hours, ETA: %.2f "
            "hours...   \r",
            100. * ratio, elapsed_hours, eta_hours);
    fflush(stderr);
    const Shape& shape = shapes[i];
    float latency = benchmark(shape);
    if (results->count(shape)) {
      (*results)[shape] = std::min(latency, (*results)[shape]);
    } else {
      (*results)[shape] = latency;
    }
  }
  fprintf(stderr, "\n");
}

int main() {
  std::map<Shape, float> results;

#ifdef GEMMLOWP_PROFILING
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif

  run_benchmarks(&results);

#ifdef GEMMLOWP_PROFILING
  gemmlowp::FinishProfiling();
#endif

  printf("Using %d thread(s)\n", kNumThreads);
  printf("depth,rows,cols,latency(s),Gop/s\n");
  for (const auto& result : results) {
    const Shape& shape = result.first;
    printf("%d,%d,%d,%.4g,%.4g\n", shape.depth, shape.rows, shape.cols,
           result.second,
           2e-9 * shape.depth * shape.rows * shape.cols / result.second);
  }
}
