<!--ts-->
   * [Continuous Integration System](#continuous-integration-system)
   * [Automated Sync from the Tensorflow Repository](#automated-sync-from-the-tensorflow-repository)
   * [Third Party GitHub Actions](#third-party-github-actions)

<!-- Added by: advaitjain, at: Wed Oct 27 03:31:51 PM PDT 2021 -->

<!--te-->

[TensorFlow repo]: https://github.com/tensorflow/tensorflow

# Continuous Integration System
  * See the [github workflow files](../.github/workflows/ci.yml) for details on
    exactly what is run as part of the GitHub Actions CI.

  * Tests can also be run from within a docker container, which can be built with:
   ```
   docker build -t tflm-ci -f ci/Dockerfile.micro .
   ```

   or use the tflm-ci docker image from [here](https://github.com/users/TFLM-bot/packages/container/package/tflm-ci).

  * You will still need to copy or mount your fork of tflite-micro on to this
    docker container prior to running any tests.

# Automated Sync from the Tensorflow Repository

While TfLite Micro and TfLite are in separate GitHub repositories, the two
projects continue to share common code.

The [TensorFlow repo][] is the single source of truth for this
shared code. As a result, any changes to this shared code must be made in the
[TensorFlow repo][] which will then automatically sync'd via a scheduled
[GitHub workflow](../.github/workflows/sync.yml).


# Third Party GitHub Actions
We use the following third party actions as part of the TFLM continuous
integration system.

 * [Create a PR](https://github.com/peter-evans/create-pull-request) to automate
   sync'ing of shared TfLite and TFLM code.

