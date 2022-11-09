import os
import subprocess
import sys

# from absl import app
# import numpy as np

# def main(_):
layers = [
    "add", "conv", "leaky_relu", "pad", "quantize", "strided_slice", "sub", "transpose_conv"]
results = {}

for layer in layers:
  results[layer] = {}
  command = str("make -f tensorflow/lite/micro/tools/make/Makefile "
                "TARGET=xtensa OPTIMIZED_KERNEL_DIR=xtensa " 
                "TARGET_ARCH=hifi4 "
                "XTENSA_TOOLS_VERSION=RI-2020.4-linux "
                "XTENSA_CORE=HIFI_190304_swupgrade "
                "test_integration_tests_seanet_" + layer + "_test -j24")
  output = os.system( command + " >> " + layer)
  file = open(layer, 'r')
  position_found = 0
  test_name = ''
  results[layer] = {}
  for line in file:
    line = str(line)
    if line.find("ALL TESTS PASSED") >= 0:
      position_found = 1
    elif position_found:
      if line.find("Testing") >= 0:
        test_name = line.split(' ')[1].strip()
      elif line.find("ticks") >= 0:
        results[layer][test_name] = line.split(' ')[2]
  os.system("rm " + layer)
  # print(results)

for k, v in results.items():
  for test_name, tick_count in v.items():
    print('{0} {1}'.format(test_name, tick_count))

  # print(results)

# if __name__ == '__main__':
#   app.run(main)
