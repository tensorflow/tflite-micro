# Magic wand example

This example shows how you can use TensorFlow Lite to run a 20 kilobyte neural
network model to recognize gestures with an accelerometer.

## Table of contents

-   [Getting started](#getting-started)
-   [Run the tests on a development machine](#run-the-tests-on-a-development-machine)
-   [Train your own model](#train-your-own-model)

To stop viewing the debug output with `screen`, hit `Ctrl+A`, immediately
followed by the `K` key, then hit the `Y` key.

## Run the tests on a development machine

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_magic_wand_test
```

This will take a few minutes, and downloads frameworks the code uses like
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, you should see a series of files get compiled, followed by some
logging output from a test, which should conclude with `~~~ALL TESTS PASSED~~~`.

If you see this, it means that a small program has been built and run that loads
the trained TensorFlow model, runs some example inputs through it, and got the
expected outputs.

## Train your own model

To train your own model, or create a new model for a new set of gestures,
follow the instructions in [magic_wand/train/README.md](train/README.md).
