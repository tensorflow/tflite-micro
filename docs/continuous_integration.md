<!--ts-->
   * [Automated Sync from the Tensorflow Repository](#automated-sync-from-the-tensorflow-repository)
   * [Third Party GitHub Actions Used](#third-party-github-actions-used)

<!-- Added by: advaitjain, at: Thu 29 Apr 2021 12:54:23 PM PDT -->

<!--te-->

[TensorFlow repo]: https://github.com/tensorflow/tensorflow

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

