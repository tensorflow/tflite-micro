#include <memory>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "flatbuffers/util.h"

/*
 * Generic model benchmark.  Evaluates runtime performance of a provided model
 * with random inputs.
 */

namespace tflm {
namespace xtensa {
namespace readfile {

namespace {

int ReadFile(const char* model_file_name) {
  std::string model_file;
  // Read the file into a string using the included util API call:
  flatbuffers::LoadFile(model_file_name, false, &model_file);
  return 0;
}

}  // namespace
}  // namespace readfile
}  // namespace xtensa
}  // namespace tflm

int main(int argc, char** argv) { return tflm::xtensa::readfile::ReadFile(argv[1]); }
