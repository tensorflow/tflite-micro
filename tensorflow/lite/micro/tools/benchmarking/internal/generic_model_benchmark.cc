#include <sys/types.h>

#include <memory>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tools/benchmarking/internal/log_utils.h"
#include "tensorflow/lite/micro/tools/benchmarking/internal/metrics.h"
#include "tensorflow/lite/micro/tools/benchmarking/internal/micro_benchmark.h"
#include "tensorflow/lite/micro/tools/benchmarking/op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "flatbuffers/util.h"

/*
 * Generic model benchmark.  Evaluates runtime performance of a provided model
 * with random inputs.
 */

namespace tflm {
namespace benchmark {

namespace {

using Profiler = ::tflite::MicroProfiler;

using TflmOpResolver = tflite::MicroMutableOpResolver<90>;

// Support all TFLM ops by default. This will use reference implementations for
// all ops unless there is an architecture specific version available.
using GenericBenchmarkRunner = MicroBenchmarkRunner<INPUT_DATA_TYPE>;

// Seed used for the random input. Input data shouldn't affect invocation timing
// so randomness isn't really needed.
constexpr int kRandomSeed = 0xF742BE52;

// Which format should be used to output debug information.
constexpr PrettyPrintType kPrintType = PrettyPrintType::kCsv;

void SetRandomInput(const int random_seed,
                    tflite::RecordingMicroInterpreter& interpreter) {
  // The pseudo-random number generator is initialized to a constant seed
  std::srand(random_seed);
  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input_tensor(i);

    // Pre-populate input tensor with random values.
    int input_length = input->bytes / sizeof(INPUT_DATA_TYPE);
    INPUT_DATA_TYPE* input_values =
        tflite::GetTensorData<INPUT_DATA_TYPE>(input);
    for (int j = 0; j < input_length; j++) {
      // Pre-populate input tensor with a random value based on a constant
      // seed.
      input_values[j] = static_cast<INPUT_DATA_TYPE>(
          std::rand() % (std::numeric_limits<INPUT_DATA_TYPE>::max() -
                         std::numeric_limits<INPUT_DATA_TYPE>::min() + 1));
    }
  }
}

int Benchmark(const char* model_file_name) {
  Profiler profiler;
  // Create an area of memory to use for input, output, and intermediate arrays.
  // Align arena to 16 bytes to avoid alignment warnings on certain platforms.
  // Tensor size is dependent on the core being used since some of them have
  // limited memory available. The TENSOR_ARENA_SIZE macro is defined in the
  // build rules.
  constexpr int kTensorArenaSize = 1024 * 1024;
  int kNumResourceVariable = 100;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  std::string model_file;
  // Read the file into a string using the included util API call:
  flatbuffers::LoadFile(model_file_name, false, &model_file);

  uint32_t event_handle = profiler.BeginEvent("TfliteGetModel");
  const tflite::Model* model = tflite::GetModel(model_file.c_str());
  profiler.EndEvent(event_handle);

  TflmOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(tflm::benchmark::CreateOpResolver(op_resolver));

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      model, op_resolver, allocator,
      tflite::MicroResourceVariables::Create(
          allocator, kNumResourceVariable),
      &profiler);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  PrettyPrintTableHeader(kPrintType, "Initialization");
  profiler.LogCsv();
  profiler.ClearEvents();

  MicroPrintf("");  // null MicroPrintf serves as a newline.

  // For streaming models, the interpreter will return kTfLiteAbort if the model
  // does not yet have enough data to make an inference. As such, we need to
  // invoke the interpreter multiple times until we either receive an error or
  // kTfLiteOk. This loop also works for non-streaming models, as they'll just
  // return kTfLiteOk after the first invocation.
  int seed = kRandomSeed;
  while (true) {
    tflm::benchmark::SetRandomInput(seed++, interpreter);
    TfLiteStatus status = interpreter.Invoke();
    if ((status != kTfLiteOk) && (status != kTfLiteAbort)) {
      MicroPrintf("Model interpreter invocation failed: %d\n", status);
      return -1;
    }

    PrettyPrintTableHeader(kPrintType, "Invocation");
    profiler.LogCsv();
    MicroPrintf("");  // null MicroPrintf serves as a newline.
    profiler.ClearEvents();

    if (status == kTfLiteOk) {
      break;
    }
  }

  LogAllocatorEvents(*allocator, kPrintType);

  return 0;
}

}  // namespace
}  // namespace benchmark
}  // namespace tflm

int main(int argc, char** argv) { return tflm::benchmark::Benchmark(argv[1]); }
