<!--
emi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
* [How to Contribute](#how-to-contribute)
   * [Contributor License Agreement](#contributor-license-agreement)
   * [Community Guidelines](#community-guidelines)
* [Code Contribution Guidelines](#code-contribution-guidelines)
   * [General Pull Request Guidelines](#general-pull-request-guidelines)
   * [Guidelines for Specific Contribution Categories](#guidelines-for-specific-contribution-categories)
      * [Bug Fixes](#bug-fixes)
      * [Reference Kernel Implementations](#reference-kernel-implementations)
      * [Optimized Kernel Implementations](#optimized-kernel-implementations)
      * [New Target / Platform / IDE / Examples](#new-target--platform--ide--examples)
* [Development Environment](#development-environment)
   * [Prerequisites](#prerequisites)
   * [Recommendations](#recommendations)
* [Development Workflow Notes](#development-workflow-notes)
   * [Before submitting your PR](#before-submitting-your-pr)
   * [During the PR review](#during-the-pr-review)
   * [Reviewer notes](#reviewer-notes)
   * [Python notes](#python-notes)
* [Continuous Integration System](#continuous-integration-system)

<!-- Added by: rkuester, at: Fri Dec 15 04:25:41 PM CST 2023 -->

<!--te-->

# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

# Code Contribution Guidelines

We provide some general guidelines with the goal of enabling community
contributions while still maintaining code health, maintainability, and
consistency in style.

Please note that while these guidelines may seem onerous to some developers,
they are derived from Google's software engineering best practices.

Before we describe project-specific guidelines, we recommend that external
contributors read these tips from the Google Testing Blog:

*   [Code Health: Providing Context with Commit Messages and Bug Reports](https://testing.googleblog.com/2017/09/code-health-providing-context-with.html)
*   [Code Health: Understanding Code In Review](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html)
*   [Code Health: Too Many Comments on Your Code Reviews?](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html)
*   [Code Health: To Comment or Not to Comment?](https://testing.googleblog.com/2017/07/code-health-to-comment-or-not-to-comment.html)

We also recommend that contributors take a look at the
[Tensorflow Contributing Guidelines](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md).

## General Pull Request Guidelines

We strongly recommend that contributors:

1.  Initiate a conversation with the TFLM team via a
    [TF Lite Micro Github issue](https://github.com/tensorflow/tensorflow/issues/new?labels=comp%3Amicro&template=70-tflite-micro-issue.md)
    as early as possible.

    *   This enables us to give guidance on how to proceed, prevent duplicated
        effort and also point to alternatives as well as context if we are not
        able to accept a particular contribution at a given time.

    *   Ideally, you should make an issue ***before*** starting to work on a
        pull request and provide context on both what you want to contribute and
        why.

1.  Once step 1. is complete and it is determined that a PR from an external
    contributor is the way to go, please follow these guidelines from
    [Google's Engineering Practices documentation](https://google.github.io/eng-practices/):

    *   [Send Small Pull Requests](https://google.github.io/eng-practices/review/developer/small-cls.html)

        *   If a pull request is doing more than one thing, the reviewer will
            request that it be broken up into two or more PRs.

    *   [Write Good Pull Request Descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html)

        *   We require that all PR descriptions link to the GitHub issue
            created in step 1 via the text `BUG=#nn` on a line by itself [^1]. This
            is enforced by CI.

            [^1]: This despite GitHub having additional forms of
            [linked references](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls).

1.  Unit tests are critical to a healthy codebase. PRs without tests should be
    the exception rather than the norm. And contributions to improve, simplify,
    or make the unit tests more exhaustive are welcome! Please refer to
    [this guideline](https://google.github.io/eng-practices/review/developer/small-cls.html#test_code)
    on how test code and writing small PRs should be reconciled.

## Guidelines for Specific Contribution Categories

We provide some additional guidelines for different categories of contributions.

### Bug Fixes

Pull requests that fix bugs are always welcome and often uncontroversial, unless
there is a conflict between different requirements from the platform, or if
fixing a bug needs a bigger architectural change.

1.  Create a [Github issue](https://github.com/tensorflow/tflite-micro/issues/new/choose)
    to determine the scope of the bug fix.
1.  Send a PR (if that is determined to be the best path forward).
1.  Bugfix PRs should be accompanied by a test case that fails prior to the fix
    and passes with the fix. This validates that the fix works as expected, and
    helps prevent future regressions.

### Reference Kernel Implementations

Pull requests that port reference kernels from TF Lite Mobile to TF Lite Micro
are welcome once we have context from the contributor on why the additional
kernel is needed.

Please see the [reference kernel porting guide](tensorflow/lite/micro/docs/porting_reference_ops.md)
for more details of that process.

### Optimized Kernel Implementations
Please see the [optimized kernel implementations guide](tensorflow/lite/micro/docs/optimized_kernel_implementations.md).

### New Target / Platform / IDE / Examples

Please see the [new platform support guide](tensorflow/lite/micro/docs/new_platform_support.md)
for documentation on how to add TFLM support for your particular platform.

# Development Environment

We support amd64-architecture development and testing on Ubuntu 22.04, although
other OSes may work.

## Prerequisites

TFLM's primary build system is [Bazel](https://bazel.build). Add
[Bazelisk](https://github.com/bazelbuild/bazelisk) as the `bazel` executable in
your PATH ([e.g., copy it to `/usr/local/bin/bazel`](ci/install_bazelisk.sh)) to
automatically download and run the correct Bazel version as specified in
`//.bazelversion`.

## Recommendations

Below are some tips that might be useful and improve the development experience.

* Add the [Refined GitHub](https://github.com/sindresorhus/refined-github)
  plugin to make the github experience even better.

* Code search the [TfLite Micro codebase](https://sourcegraph.com/github.com/tensorflow/tflite-micro@main)
  on Sourcegraph. And optionally install the [plugin that enables GitHub integration](https://docs.sourcegraph.com/integration/github#github-integration-with-sourcegraph).

* Install
  [Buildifier](https://github.com/bazelbuild/buildtools/blob/master/buildifier/README.md)
  ([e.g.](ci/install_buildifier.sh)) to format Bazel BUILD and .bzl files.

* Install the latest clang and clang-format. For example, [here](ci/Dockerfile.micro)
  is the what we do for the TFLM continuous integration Docker container.

* Get a copy of [cpplint](https://github.com/google/styleguide/tree/gh-pages/cpplint)
  or install it:

* Install Pillow.  For example, [here](ci/Dockerfile.micro) is what we do for
  the TFLM continuous integration Docker container.

  ```
  pip install cpplint
  ```

* [yapf](https://github.com/google/yapf/) should be used for formatting Python
  code. For example:

  ```
  pip install yapf
  yapf log_parser.py -i'
  ```

* Add a git hook to check for code style etc. prior to creating a pull request:
  ```
  cp tensorflow/lite/micro/tools/dev_setup/pre-push.tflm .git/hooks/pre-push
  ```

# Development Workflow Notes

## Before submitting your PR

1.  Run in-place clang-format on all the files that are modified in your git
    tree with

    ```
    clang-format -i -style=google `git ls-files -m | grep "\.cc"`
    clang-format -i -style=google `git ls-files -m | grep "\.h"`
    ```

1.  Make sure your code is lint-free.

    ```
    cpplint `git ls-files -m`
    ```

1.  Run all the tests for x86, and any other platform that you are modifying.

    ```
    tensorflow/lite/micro/tools/ci_build/test_x86_default.sh
    ```

    Please check the READMEs in the optimized kernel directories for specific
    instructions.

1.  Sometimes, bugs are caught by the sanitizers that can go unnoticed
    via the Makefile. To run a test with the different sanitizers, use the
    following commands (replace `micro_interpreter_test` with the target that you
    want to test:

    ```
    CC=clang bazel run --config=asan tensorflow/lite/micro:micro_interpreter_test
    CC=clang bazel run --config=msan tensorflow/lite/micro:micro_interpreter_test
    CC=clang bazel run --config=ubsan tensorflow/lite/micro:micro_interpreter_test
    ```

## During the PR review

1.  Do not change the git version history.

    *   Always merge upstream/main (***do not rebase***) and no force-pushes
        please.

    *   Having an extra merge commit is ok as the github review tool handles
        that gracefully.

    Assuming that you forked tensorflow and added a remote called upstream with:

    ```
    git remote add upstream https://github.com/tensorflow/tflite-micro.git
    ```

    Fetch the latest changes from upstream and merge into your local branch.

    ```
    git fetch upstream
    git merge upstream/main
    ```

    In case of a merge conflict, resolve via:

    ```
    git mergetool

    # Use your favorite diff tools (e.g. meld) to resolve the conflicts.

    git add <files that were manually resolved>

    git commit
    ```

1.  If a force push seems to be the only path forward, please stop and let your
    PR reviewer know ***before*** force pushing. We will attempt to do the merge
    for you. This will also help us better understand in what conditions a
    force-push may be unavoidable.

## Reviewer notes

*   [GIthub CLI](https://cli.github.com) can be useful to quickly checkout a PR
    to test locally.

    `gh pr checkout <PR number>`

*   Google engineers on the Tensorflow team will have the permissions to push
    edits to most PRs. This can be useful to make some small fixes as a result
    of errors due to internal checks that are not easily reproducible via
    github.

    One example of this is
    [this comment](https://github.com/tensorflow/tensorflow/pull/38634#issuecomment-683190474).

    And a sketch of the steps:

    ```
    git remote add <remote_name> git@github.com:<PR author>/tflite-micro.git
    git fetch <remote_name>

    git checkout -b <local-branch-name> <remote_name>/<PR branch name>

    # make changes and commit to local branch

    # push changes to remove branch

    git push <remote_name> <PR branch name>

    # remove the temp remote to clean up your git environment.

    git remote rm <remote_name>
    ```

## Python notes

*   [TFLM Python guide](docs/python.md)

# Continuous Integration System
  * Some [additional documentation](docs/continuous_integration.md) on the TFLM CI.
