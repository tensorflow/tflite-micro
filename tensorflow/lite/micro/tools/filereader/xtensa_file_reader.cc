#include <memory>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "stdio.h"

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
  FILE *fp = fopen(model_file_name, "r");
  char filecontent[1000];
  while (fgets(filecontent, 1000, fp) != nullptr){
    model_file = model_file.append(filecontent);
  }
  printf("%s", model_file.c_str());
  fclose(fp);
  return 0;
}

}  // namespace
}  // namespace readfile
}  // namespace xtensa
}  // namespace tflm

int main(int argc, char** argv) { return tflm::xtensa::readfile::ReadFile(argv[1]); }
