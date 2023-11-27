# C++ Layer by Layer Output Tool on TFLite Micro

This tool allows you to pass in a TfLite model and returns a flatbuffer file with input and the corresponding output values appended into it that can be passed into a python debugging tool which can compare those golden values vs the x86 TFLM reference kernel implementation.

The C++ Tool/binary will write a debugging flatbuffer to the path provide in 2nd arg
using the tflite_model provided in the 1st arg 

##### Command bazel/blaze:

```
 bazel run tensorflow/lite/micro/tools:layer_cc -- \
    </path/to/input_model.tflite>
   </path/to/output.file_name>
```

##### How to Build using Makefile :

```
make -f tensorflow/lite/micro/tools/make/Makefile layer_by_layer_output_tool -j24
```

# Python Layer by Layer Debugging Tool 

The Python Tool/Script can first be used to comapre TFLM vs Tflite outputs for random inputs by only providing a TfLite file.

#### TfLite vs TFLM command:
``` 
 bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
    --input_tflite_file=</path/to/my_model.tflite>
```

The Python Tool/Script can also be used to compare, TFLM's python x86 output vs expected output provided by the C++ Tool/binary. 

#### TFLM vs Expected Command:
``` 
  bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
    --input_tflite_file=</path/to/my_model.tflite> \
    --layer_by_layer_data_file=</path/to/my_debug_flatbuffer_file>
```

#### Optional Flags:
 ` --print_dump  `
When this flag is set, it will print the TFLM output for each layer that is compared.

 ` --rng`
Integer random number seed for generating input data for comparisons against TFLite. (Default: 42)
