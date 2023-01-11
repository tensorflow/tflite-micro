// A simple program to test building against the Tensorflow library shipping in
// the Tensorflow Python package.

#include <tensorflow/core/util/util.h>

int main(int argc, char* argv[]) {
    const char* ptr = "test";
    const size_t n = 4;
    tensorflow::PrintMemory(ptr, n);
    return 0;
}
