# Generate Micro Mutable Op Resolver from a model

Using the AllOpsResolver will link all the TFLM operators into the executable, which will add significantly to the memory footprint.
Using the MicroMutableOpResolver will include the operators specified by the user. This generally requires manually finding out which operators are used in the model through the use of a visualization tool, which may be impractical in some cases.
This script will automatically generate a MicroMutableOpResolver with only the used operators for a given model.

## How to run

bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --input_tflite_file=<path to tflite file> --output_dir=<output directory>

Note that final output directory will be <output directory>/<base name of model>.

Example:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --input_tflite_file=/tmp/person_detect.tflite --output_dir=/tmp/tflite-micro/gen_dir
```

A header file called, gen_micro_mutable_op_resolver.h will be created in /tmp/gen_dir/person_detect.

This can then be included in the application and used like below:

```
tflite::MicroMutableOpResolver<kNumberOperators> op_resolver = get_resolver();
```

## How to test

Another script can be used to verify the generated header file. This will also demonstrate the usage of generated header file.

bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=<path to tflite file> --output_dir=<output directory>

Note that final output directory will be <output directory>/<base name of model>.

Example:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=/tmp/person_detect.tflite --output_dir=/tmp/tflite-micro/gen_dir
```

Note that the generated test must be somewhere in tflite-micro and it also need to be run from there.
Also note that Bazel expects absolute paths.

Example:

```
cd /tmp/tflite-micro
bazel run gen_dir/person_detect:micro_mutable_op_resolver_test
```

### Verifying the output of a given model

By default the model will run without any generated input or verifying the output. This can be done by adding the flag --verify_output=1.

Example:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --input_tflite_file=/tmp/my_model.tflite --output_dir=$(realpath gen_dir)
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=/tmp/my_model.tflite --output_dir=$(realpath gen_dir) --verify_output=1
bazel run gen_dir/my_model:micro_mutable_op_resolver_test

```

Depending on the size of the input model the arena size may need to be increased. Arena size can be set with --arena_size=<size>.

Example:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --input_tflite_file=/tmp/big_model.tflite --output_dir=gen_dir
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=/tmp/big_model.tflite --output_dir=gen_dir --verify_output=1 --arena_size=1000000
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generated/big_model:micro_mutable_op_resolver_test

```





