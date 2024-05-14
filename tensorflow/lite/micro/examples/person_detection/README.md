# Person detection example

This example shows how you can use Tensorflow Lite to run a 250 kilobyte neural
network to recognize people in images.

## Table of contents

-   [Run the tests on a development machine](#run-the-tests-on-a-development-machine)
-   [Training your own model](#training-your-own-model)
-   [Additional makefile targets](#additional-makefile-targets)


## Run the tests on a development machine

```
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
make -f tensorflow/lite/micro/tools/make/Makefile test_person_detection_test
```

You should see a series of files get compiled, followed by some logging output
from a test, which should conclude with `~~~ALL TESTS PASSED~~~`. If you see
this, it means that a small program has been built and run that loads a trained
TensorFlow model, runs some example images through it, and got the expected
outputs. This particular test runs images with a and without a person in them,
and checks that the network correctly identifies them.

To understand how TensorFlow Lite does this, you can look at
[person_detection_test.cc](person_detection_test.cc).

## Additional makefile targets
```
make -f tensorflow/lite/micro/tools/make/Makefile person_detection
make -f tensorflow/lite/micro/tools/make/Makefile person_detection_bin
make -f tensorflow/lite/micro/tools/make/Makefile run_person_detection
```

The `run_person_detection` target will produce continuous output similar
to the following:
```
person score:-72 no person score 72
```

## Training your own model

You can train your own model with some easy-to-use scripts. See
[training_a_model.md](training_a_model.md) for instructions.
