<!-- mdformat off(b/169948621#comment2) -->

# DP-Net Example 

This example loads the model: 
```bash
tensorflow/lite/micro/examples/dp_net/dp_net.tflite
```

The model can be updated by replacing the file with a new version of the same name and recompiling the application.

## Compile Example
```bash
make -f tensorflow/lite/micro/tools/make/Makefile dp_net 
```
* Note: first compilation downloads dependencies and builds TFLM library, subsequent runs will only compiled changed files.

## Run dp_net example
```bash
./tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/bin/dp_net 
```
* Note: currently fails on REVERSE_V2 op which is not supported by TFLM
