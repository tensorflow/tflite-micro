<!-- mdformat off(b/169948621#comment2) -->

# add_model_float example

This example loads the model: 
```bash
tensorflow/lite/micro/examples/dp_net/add_model_float.tflite
```

Currently this model performs f(x) = x + x

The model can be updated by replacing the file with a new version of the same name and recompiling the application.

## Compile example
```bash
make -f tensorflow/lite/micro/tools/make/Makefile add_model_float 
```

## Compile example Test
```bash
make -f tensorflow/lite/micro/tools/make/Makefile add_model_float_test
```

* Note: first compilation downloads dependencies and builds TFLM library, subsequent runs will only compile changed files.

## Run add_model_float example
```bash
./tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/bin/add_model_float
```

## Run add_model_float test 
```bash
./tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/bin/add_model_float_test
```
