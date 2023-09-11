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

This is just to test the actual script that generates the micro mutable ops resolver header for a given model.
So that the actual list of operators corresponds to a given model and that the syntax of the header is correct.

For this another script can be used to verify the generated header file:

```
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=<path to tflite file> --output_dir=<output directory>
```

This script verifies a single model at a time. It will generate a small inference testing app that is using the generated header file, which can then be executed and tested as a final step.
Because of this the specified output path will be appended with the name of the model so that the generated test is named after the model.
In other words the final output directory will be <output directory>/<base name of model>.

The essence of this is that different output paths need to be specified for the actual header script and the actual test script.

So there will be 3 steps,
1) Generate the micro mutable specifying e.g. output path gen_dir/<base_name_of_model>
2) Generate the micro mutable specifying e.g. output path gen_dir
3) Run the generated test

Example assuming /tmp/my_model.tflite exists:

```
# Step 1 generates header to gen_dir/my_model
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --common_tflite_path=/tmp/ \
             --input_tflite_files=my_model.tflite --output_dir=$(realpath gen_dir/my_model)

# Step 2 generates test app using header from step 1 to gen_dir/my_model since my my_model is appended
bazel run tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=/tmp/my_model.tflite --output_dir=$(realpath gen_dir) --verify_output=1

# Step 3 runs the generated my_model test
bazel run gen_dir/my_model:micro_mutable_op_resolver_test

```

Note1: Bazel expects absolute paths.
Note2: By default the inference model test will run without any generated input or verifying the output. Verifying output can be done with --verify_output=1, which is done in the example above.
Note3: Depending on the size of the model the arena size may need to be increased. Arena size can be set with --arena_size=<size>.
