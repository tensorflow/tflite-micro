<!-- Define reference-style links used throughout the document -->
[small PRs]: https://google.github.io/eng-practices/review/developer/small-cls.html
[Micro Contributing Guidelines]: https://github.com/tensorflow/tflite-micro/blob/main/CONTRIBUTING.md
[Providing Context]: https://testing.googleblog.com/2017/09/code-health-providing-context-with.html
[`ParseOpDataTfLite()`]: https://github.com/tensorflow/tensorflow/blob/d8394a6d774f5e3c02d97f1fc18ff445199db598/tensorflow/lite/core/api/flatbuffer_conversions.cc#L135
[PR #45307]: https://github.com/tensorflow/tensorflow/pull/45307
[PR #46021]: https://github.com/tensorflow/tensorflow/pull/46021
[PR #45311]: https://github.com/tensorflow/tensorflow/pull/45311
[PR #45457]: https://github.com/tensorflow/tensorflow/pull/45457
[PR #45646]: https://github.com/tensorflow/tensorflow/pull/45646
[PR #45647]: https://github.com/tensorflow/tensorflow/pull/45647
[pre-submit checklist]: https://github.com/tensorflow/tflite-micro/blob/main/CONTRIBUTING.md#before-submitting-your-pr
[reference_ops.h]: https://github.com/tensorflow/tensorflow/blob/92f459e6b917fa5099ef5317d14c5100d33a86f0/tensorflow/lite/kernels/internal/reference/reference_ops.h
[general porting guidelines]: #general-porting-guidelines

# Porting Reference Ops from Lite to Micro

This is a guide to porting reference ops from Lite to Micro. It explains,
step-by-step, the recommended code changes and the process for submitting them
for review and acceptance.  The process results in multiple pull requests, or
PRs. Multiple, [small PRs][] are easier for the project to review and merge.

The [Micro Contributing Guidelines][] are prerequisite reading. They cover
general code health, maintainability, style, and submission, as well as how to
setup a development environment. This guide contains step-by-step instructions
for the specific task of porting reference ops from Lite to Micro.

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->
<!--ts-->
   * [Porting Reference Ops from Lite to Micro](#porting-reference-ops-from-lite-to-micro)
      * [1. Look for a port already in progress](#1-look-for-a-port-already-in-progress)
      * [2. Open a GitHub issue to track the port](#2-open-a-github-issue-to-track-the-port)
      * [3. Extract Lite's code for parsing op parameters to a function (PR1)](#3-extract-lites-code-for-parsing-op-parameters-to-a-function-pr1)
      * [4. Extract the reference for the op to a standalone header (PR2)](#4-extract-the-reference-for-the-op-to-a-standalone-header-pr2)
      * [5. Port the op from Lite to Micro (PR3)](#5-port-the-op-from-lite-to-micro-pr3)
   * [General Guidelines](#general-guidelines)
      * [Check each commit for formatting, lint, and unit-test passage](#check-each-commit-for-formatting-lint-and-unit-test-passage)
      * [Maintain a 1:1 correspondence between Micro and Lite versions of unit tests](#maintain-a-11-correspondence-between-micro-and-lite-versions-of-unit-tests)
   * [Notes](#notes)
   * [Frequently Asked Questions](#frequently-asked-questions)
      * [Can I use malloc/free or new/delete in my operator code?](#can-i-use-mallocfree-or-newdelete-in-my-operator-code)
      * [Can I use static variable allocation in my operator code?](#can-i-use-static-variable-allocation-in-my-operator-code)
      * [How do I allocate persistent memory?](#how-do-i-allocate-persistent-memory)
      * [When am I allowed to allocate persistent memory?](#when-am-i-allowed-to-allocate-persistent-memory)
      * [How do I allocate/use temporary memory?](#how-do-i-allocateuse-temporary-memory)
      * [When can I allocate/use temporary memory?](#when-can-i-allocateuse-temporary-memory)
      * [Can I resize my input/output tensors?](#can-i-resize-my-inputoutput-tensors)
      * [Can I change the shape of tensors in my operator code?](#can-i-change-the-shape-of-tensors-in-my-operator-code)
      * [When can I change the shape of tensors in my operator code?](#when-can-i-change-the-shape-of-tensors-in-my-operator-code)
      * [Can I modify a TfLiteTensor or TfLiteEvalTensor?](#can-i-modify-a-tflitetensor-or-tfliteevaltensor)

<!-- Added by: advaitjain, at: Thu 16 Sep 2021 11:49:51 AM PDT -->

<!--te-->

## 1. Look for a port already in progress

Begin by searching the tflite-micro GitHub repository for issues containing the
name of the op under consideration to ensure someone isn't already working on a
port.

## 2. Open a GitHub issue to track the port

Open a GitHub issue to announce your intent to port the op, and to begin a
record of your work. Document the entire process of porting the op in this
issue. Link constituent PRs to this issue. See the article [Providing
Context][] for background on documenting your work via bug reports.

## 3. Extract Lite's code for parsing op parameters to a function (PR1)

Now we begin changing, testing, and submitting code. This step will result in
the first pull request, PR1.

1.  Extract the code for parsing op parameters out of the switch statement in
    [`ParseOpDataTfLite()`][] in `lite/core/api/flatbuffer_conversions.cc` into
    a standalone function, and call that function from the switch statement.
    This standalone function is now available to be called by the Micro op
    resolver, which also needs to parse the op parameters, in a future change.
    A simple example is [PR #45307][], and a more complicated example is [PR
    #46021][].

1.  Use `clang-format` to make sure the code is properly formatted.

    ```shell
    clang-format --style=google -i $(git ls-files -m | grep -E '\.cc|\.h')
    ```

1.  Make sure your code is lint-free.

    ```shell
    cpplint.py $(git ls-files -m)
    ```

1.  Create a single commit containing the change. Observe the guidelines for
    good commit log messages found in the article [Providing Context][].
    A good example is commit [0664214](https://github.com/tensorflow/tensorflow/pull/45307/commits/0664214792ad2357f6224e7002661894775cb512).

1.  Since this change modifies the op's implementation in Lite, test the change
    with the relevant Lite unit tests.

    ```shell
    bazel test tensorflow/lite/kernels:all
    ```

1.  Create and submit the PR. Write a [good PR description][], and be sure to
    link to the GitHub issue created to document the port. A good example is
    [PR #45307][].

    [good PR description]: https://google.github.io/eng-practices/review/developer/cl-descriptions.html

## 4. Extract the reference for the op to a standalone header (PR2)

Move the reference implementation of the op in [reference_ops.h][] to a standalone header so that
Micro can include it without including unrelated dependencies via
reference_ops.h.

A good example is [PR #45311][].

1.  Copy an existing header from `tensorflow/lite/kernels/internal/reference/`
    to `tensorflow/lite/kernels/internal/reference/NEW_OP.H` to create the
    boilerplate. Replace `NEW_OP.H` with the name of the new operator.

1.  Move the implementation from
    `tensorflow/lite/kernels/internal/reference/reference_ops.h` to
    `tensorflow/lite/kernels/internal/reference/NEW_OP.H`.

1.  Add the new header to the build by adding to the  library definitions
    `reference_base` and `legacy_reference_base` in the file
    `tensorflow/lite/kernels/internal/BUILD`. See, for example,
    [this change for operator FILL](https://github.com/tensorflow/tensorflow/pull/45311/commits/92f459e6b917fa5099ef5317d14c5100d33a86f0#diff-0b0fc9e1affece3c5a141ee9326f882876b6b958bc8b12a7c01d7540dc04983e).

1.  Use the program `clang-format` to make sure the code is properly formatted.

    ```shell
    clang-format --style=google -i $(git ls-files -m | grep -E '\.cc|\.h')
    ```

    Do not clang-format existing code in `BUILD` or `reference_ops.h`.

1.  Make sure your code is lint-free.

    ```shell
    cpplint.py $(git ls-files -m)
    ```

    Do not modify code in `BUILD` or `reference_ops.h` to satisfy `cpplint.py`.

1.  Create a single commit containing the change. Observe the guidelines for
    good commit log messages found in the article [Providing Context][].
    A good example is commit [92f459e](https://github.com/tensorflow/tensorflow/commit/92f459e6b917fa5099ef5317d14c5100d33a86f0).

1.  Since this change modifies the op's implementation in Lite, test the change
    with the relevant Lite unit tests.

    ```shell
    bazel test tensorflow/lite/kernels:all
    ```

1.  Create and submit the PR. Write a [good PR description][], and be sure to
    link to the GitHub issue created to document the port. A good example is
    [PR #45311][].

## 5. Port the op from Lite to Micro (PR3)

1.  Copy the kernel and test from Lite to Micro.

    In the first commit of this PR, copy the kernel and test from Lite to Micro
    without making any modifications and without adding them to the build.

    A good example is commit [a2ca1fd](https://github.com/tensorflow/tensorflow/commit/a2ca1fd7a174438f736c0435dd3e4e618612fdee).

    This copy action is in its own commit in order to create readable, reviewable diffs
    when modifications are made in later commits. If the files were copied and
    modified in one step, the modifications would not appear as a diff of the Lite
    version. Instead, the files would simply appear at the destination path in
    their final form.


1.  Remove Lite-specific code from copies

    In the second commit of this PR, remove the bulk of Lite-specific code from
    the files copied to micro in the previous step.

    A good example is commit [a5a87b4](https://github.com/tensorflow/tensorflow/commit/a5a87b420b87a1f832e241db3a5b724207ea700a).

    This bulk-delete action is in its own commit for reasons similar to
    those given in the step above: to produce a more readable, reviewable diff in this
    step and in the next. Because the files are not yet added to the build, they
    need not (and obviously won't) compiler or function. What to delete now as
    opposed to deleting in the next commit is somewhat subjective, but make
    deletes in order to:

    -   Flatten the namespace down to `tflite`.
    -   Stop resizing output tensors.
    -   Remove input and output types other than `int8`, `int16`, and `float32`.
    -   Stop using gmock and gtest.
    -   etc.

1.  Port the op and the test

    Make the necessary changes to the micro kernel, header, and test to make the op
    implementation suitable for micro. Include these in the build.

    This step requires the most creativity, and may receive the most feedback
    during review. Maintain good atomicity in your commits. Considering its
    scope, this step will consist of more than one commit. A good example is
    the changes made in [PR #45647][].

1.  Use `clang-format` to make sure the code is properly formatted.

    ```shell
    $ clang-format --style=google -i $(git ls-files -m | grep -E '\.cc|\.h')
    ```

    Do not clang-format existing code in `BUILD` or `reference_ops.h`.

1.  Make sure the code is lint-free.

    ```shell
    $ cpplint.py $(git ls-files -m)
    ```

    Do not modify code in `BUILD` or `reference_ops.h` to satisfy `cpplint.py`.

1.  Make sure the port passes all applicable tests.

    ```shell
    $ bazel test tensorflow/lite/micro/kernels:${op}_test
    $ bazel test tensorflow/lite/micro/kernels:all
    $ make -f tensorflow/lite/micro/tools/make/Makefile test_kernel_${op}_test
    $ make -f tensorflow/lite/micro/tools/make/Makefile test
    ```

    See the general [Micro Contributing Guidelines][] for other testing ideas,
    including the use of address sanitizers.

1.  Create and submit the PR. Write a [good PR description][], and be sure to
    link to the GitHub issue created to document the port. A good example is
    [PR #45647][].

# General Guidelines

## Check each commit for formatting, lint, and unit-test passage

Check each commit against the [pre-submit checklist][] in the micro
Contributing Guidelines. Specifically, make sure your code:

1.  Is formatted with clang-format.
1.  Passes a lint check.
1.  Passes all unit tests.

    ```shell
    $ make -s -j8 -f tensorflow/lite/micro/tools/make/Makefile test
    ```

CI runs these checks on all PRs, and will hold up your PR if any of these checks fail.

## Maintain a 1:1 correspondence between Micro and Lite versions of unit tests

To the extent possible, maintain a 1:1 correspondence between Micro and Lite
versions of unit tests. Avoid cleanup of merely stylistic issues, e.g., by
replacing the hardcoded literal `3.40282e+038` with
`std::numeric_limits<float>::max()`. Any changes between the Micro and Lite
versions of a test put a burden on future maintainers to figure out whether the
differences are actually significant or just stylistic.

# Notes

*   There was discussion of commits vs. PRs in [#45387](https://github.com/tensorflow/tensorflow/issues/45387).

*   [TensorFlow Lite 8-bit quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec)

# Frequently Asked Questions

## Can I use malloc/free or new/delete in my operator code?
No.  All memory allocation in TensorFlow Lite Micro (TFLM) is done using C++
stack based automatic allocation, or through specialized TFLM persistent
and temporary allocation methods.

## Can I use static variable allocation in my operator code?
No.  This is due to the call ordering of C++ static constructors being
platform/compiler dependent.

## How do I allocate persistent memory?
Use `TfLiteContext::AllocatePersistentBuffer` to allocate persistent memory.
Memory allocated by this method will remain valid throughout the lifetime of
the `tflite::MicroInterpreter` instance.

An example code snippet looks like ([leaky_relu.cc](../kernels/leaky_relu.cc)):
```C++
void* LeakyReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(LeakyReluOpData));
}
```

## When am I allowed to allocate persistent memory?
The `TfLiteContext::AllocatePersistentBuffer` method may only be called within
the scope of your operator's `Init` and `Prepare` methods.

## How do I allocate/use temporary memory?
Use the `TfLiteContext::RequestScratchBufferInArena` and
`TfLiteContext::GetScratchBuffer` methods.  The temporary memory is shared
between all operators, and is only valid for your operator within the scope
of your operator's `Invoke` method.  Do not attempt to use temporary memory
to share data between operator invocations.  Temporary memory is to be used
only as pre-allocated storage during the execution scope of your operator's
`Invoke` method.

An example code snippet looks like ([add_n.cc](../kernels/add_n.cc)):
```C++
if (output->type == kTfLiteFloat32) {
    // Allocate scratch buffer space for pointer to each tensor's data
    // and store the scratch buffer index in the node's user_data
    int scratch_index;
    size_t scratch_size = sizeof(float*) * num_inputs;
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, scratch_size, &scratch_index));
    node->user_data =
        reinterpret_cast<decltype(node->user_data)>(scratch_index);
  }
```
And to use the buffer:
```C++
int scratch_index =
    static_cast<int>(reinterpret_cast<intptr_t>(node->user_data));
void* scratch_buffer = context->GetScratchBuffer(context, scratch_index);
```

## When can I allocate/use temporary memory?
The `TfLiteContext::RequestScratchBufferInArena` method is available only within
the scope of your operator's `Prepare` method.
The `TfLiteContext::GetScratchBuffer` method is available only within
the scope of your operator's `Invoke` method.

## Can I resize my input/output tensors?
No.  The storage space for each input/output tensor is a fixed, calculated value
determined at the time the TensorFlow Lite (TfLite) model converter is executed.
During the `Init` phase of the `tflite::MicroInterpreter` all tensor storage is
allocated by the `tflite::MicroInterpreter` instance, using the calculated values
of the model converter.
For more information see: [Memory Allocation Overview](online_memory_allocation_overview.md)

## Can I change the shape of tensors in my operator code?
Yes.  The new shape must not exceed the storage space indicated by the old shape.
Because tensor shape values may live in memory that is not directly writable
(ex. Flash, EEPROM, ROM), a special method must be called before modification
is attempted.  The `tflite::micro::CreateWritableTensorDimsWithCopy` method will
move the tensor shape values to guaranteed persistent writable memory.

An example code snippet looks like ([l2_pool_2d.cc](../kernels/l2_pool_2d.cc)):
```C++
// the output variable is a TfLiteTensor*
TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                               context, output, output_eval));
output->dims->data[kBatchRank] = batches;
output->dims->data[kHeightRank] = out_height;
output->dims->data[kWidthRank] = out_width;
output->dims->data[kChannelRank] = channels_out;
```

## When can I change the shape of tensors in my operator code?
Tensor shape values can be modified any time after the
`tflite::micro::CreateWritableTensorDimsWithCopy` method has been called.
This means that tensor shape values can be modified within the scope of
your operator's `Prepare` or `Invoke` methods.
The `tflite::micro::CreateWritableTensorDimsWithCopy` method may
only be called within the scope of your operator's `Prepare` method.

## Can I modify a `TfLiteTensor` or `TfLiteEvalTensor`?
No.  The `tflite::MicroInterpreter` is the owner and manipulator of these data
structures.  Your code should not modify these data structures.  The only
directly allowed modification of tensors is to change their data values, or
their shape values.

## How do I fix optimized kernel unit test failures?
Kernel unit tests for all optimizated kernels should pass. By default kernel unit 
tests for the newly added op may fail for optimized kernels as they may not have the
correct references. In this case, we should let the optimized kernels fall back
to the newly added reference kernels. For example, refer to this [this commit](https://github.com/tensorflow/tflite-micro/pull/1274/commits/d36c9dd598dcbf352f2c60463fd0d4153703a1cd).
