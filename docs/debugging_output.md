# How to debug invalid output

The TFLM debugging output tools allow TFLM users to easily debug their models
by providing a tool that will compare the intermediate  values(output of each OP/Kernel)
from a model post invoke between the TFLM and TfLite. As well as a way to
compare intermediate values between TFLM x86 implementations and Optimized 
Implementations.

## How to debug TFLM Interpreter output on embedded targets

First you call a C++ binary that takes a TfLite model and returns a file that has
random inputs and their corresponding output values for each layer of the model
it was provided.

The second is you provide a TfLite model and file outputted by C++ binary above
to a  python script. The script runs TFLM x86 inference comparison to the 
expected output.

## How to debug TFLM Python Interpreter output

Using a python script mentioned in the section above when only a TfLite model is
provided as input, the script generates random input and compares TFLM vs TfLite
inference outputs for each layer of the model.

## C++ Expected Layer by Layer Output Tool on TFLite Micro

This C++ binary allows you to pass in a TfLite model and returns a flatbuffer
file with input and the corresponding output values appended into it that can be
passed into a python debugging tool which can compare those golden values vs
the x86 TFLM reference kernel implementation.

The C++ Tool/binary will write a debugging file to the path provide in
2nd arg using the tflite_model provided in the 1st arg.

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

## Python Layer by Layer Debugging Tool 

The Python Tool/Script can first be used to compare TFLM vs Tflite outputs for
random inputs by only providing a TfLite file.

#### TfLite vs TFLM command:
``` 
 bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
    --input_tflite_file=</path/to/my_model.tflite>
```

The Python Tool/Script can also be used to compare TFLM's python x86 output
vs expected output provided by the C++ Tool/binary.

#### TFLM vs Expected Command:
``` 
  bazel run tensorflow/lite/micro/tools:layer_by_layer_debugger -- \
    --input_tflite_file=</path/to/my_model.tflite> \
    --layer_by_layer_data_file=</path/to/my_debug_flatbuffer_file>
```

#### Optional Flags:
 ` --print_dump  `
When this flag is set, it will print the TFLM output for each layer that is
compared.

 ` --rng`
Integer random number seed for generating input data for comparisons against TFLite. (Default: 42)
