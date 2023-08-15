# Codegen Hello World Example

This is a code-generated example of the hello world model.

To generate the inference code at `codegen/example/hello_world_model.h/.cc`:

```
bazel run codegen:code_generator -- \
  --model $(pwd)/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite \
  --output_dir $(pwd)/codegen/examples/hello_world \
  --output_name hello_world_model
```

To compile the generated source, you can use the Makefile:

```
make -f tensorflow/lite/micro/tools/make/Makefile codegen_hello_world
```
