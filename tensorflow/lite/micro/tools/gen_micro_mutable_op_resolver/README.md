# Generate Micro Mutable Op Resolver from a model

The MicroMutableOpResolver includes the operators explictly specified in source code.
This generally requires manually finding out which operators are used in the model through the use of a visualization tool, which may be impractical in some cases.
This script will automatically generate a MicroMutableOpResolver with only the used operators for a given model or set of models.

Note: Check ci/Dockerfile.micro for supported python version.

## How to run

bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --common_tflite_path=<path to tflite file> \
             --input_tflite_files=<name of tflite file(s)> --output_dir=<output directory>

Note that if having only one tflite as input, the final output directory will be <output directory>/<base name of model>.

Example:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --common_tflite_path=/tmp/model_dir \
             --input_tflite_files=person_detect.tflite --output_dir=/tmp/gen_dir
```

A header file called, gen_micro_mutable_op_resolver.h will be created in /tmp/gen_dir/person_detect.

Example:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --common_tflite_path=/tmp/model_dir \
             --input_tflite_files=person_detect.tflite,keyword_scrambled.tflite --output_dir=/tmp/gen_dir
```
A header file called, gen_micro_mutable_op_resolver.h will be created in /tmp/gen_dir.

Note that with multiple tflite files as input, the files must be placed in the same common directory.

The generated header file can then be included in the application and used like below:

```
tflite::MicroMutableOpResolver<kNumberOperators> op_resolver = get_resolver();
```

## Verifying the content of the generated header file

Run the unit test that generates the resolver header for `person_detect.tflite` and verifies model invocation:

```
bazel test //tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:micro_mutable_op_resolver_test
```
