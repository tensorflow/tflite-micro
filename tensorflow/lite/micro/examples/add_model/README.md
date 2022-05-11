<!-- mdformat off(b/169948621#comment2) -->

# add_model example

This example loads the model: 
```bash
tensorflow/lite/micro/examples/dp_net/add_model.tflite
```

Currently this model performs f(x) = x + x

The model can be updated by replacing the file with a new version of the same name and recompiling the application.

## Compile example
```bash
make -f tensorflow/lite/micro/tools/make/Makefile add_model 
```

## Compile example Test
```bash
make -f tensorflow/lite/micro/tools/make/Makefile add_model_test
```

* Note: first compilation downloads dependencies and builds TFLM library, subsequent runs will only compile changed files.

## Run add_model example
```bash
./tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/bin/add_model
```

## Run add_model test 
```bash
./tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/bin/add_model_test
```
