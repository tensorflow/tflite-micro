# Codegen Hello World Example

This is a code-generated example of the hello world model. The process is
currently somewhat involved:

## Build the preprocessor for your target

This creates a target-specific preprocessor binary capable of performing the
init and prepare stages of the Interpreter and serializing the output. This
binary can be re-used for multiple models.

### x86
```
make -f tensorflow/lite/micro/tools/make/Makefile codegen_preprocessor
```

## Run the preprocessor

The preprocessor will take the provided model, create a TFLM Interpreter, and
allocate tensors. It will then capture and serialize the resulting data
structures needed for inference. For embedded targets, this should be run under
simulation.

### x86
```
./gen/linux_x86_64_default/bin/codegen_preprocessor \
  $(pwd)/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite \
  $(pwd)/gen/linux_86_64_default/genfiles/hello_world_int8.ppd
```

## Generate the inference code

To generate the inference code at `codegen/example/hello_world_model.h/.cc`:

### x86
```
bazel run codegen:code_generator -- \
  --model $(pwd)/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite \
  --preprocessed_data $(pwd)/gen/linux_86_64_default/genfiles/hello_world_int8.ppd \
  --output_dir $(pwd)/codegen/examples/hello_world \
  --output_name hello_world_model
```

## Compile the generated inference code

 To compile the generated source, you can use the Makefile:

### x86
```
make -f tensorflow/lite/micro/tools/make/Makefile codegen_hello_world
```
