# Continuous Integration Docs Contents
* [User Facing CI](#user-facing-ci)
  * [`ci:run` - Run Tests](##`ci:run`)
  * [`ci:ready_to_merge` - Send To Merge Queue](#ci:ready_to_merge)
* [Manually Running Tests](#manually-running-tests)
* [Sync From Tensorflow Repository](#sync-from-the-tensorflow-repository)
* [Merge Queue Details](#merge-queue-details)

# User Facing CI
The continuous integration system is controlled by applying labels to PRs. There are only two important labels: `ci:run` runs the testing suite, and `ci:ready_to_merge` places a PR in the merge queue.
  ## `ci:run`
  The `ci:run` label runs the [main testing suite](../.github/workflows/ci.yml) against the PR. For details of the tests involved, examine the linked file. The `ci:run` tag is self-removing.
  ## `ci:ready_to_merge`
  After all tests from `ci:run` have passed, the Google CLA has been agreed to, and a reviewer has approved the PR, applying the `ci:ready_to_merge` label will enter the PR into the merge queue. Unless there is a conflict with other PR's in the queue, this should be a fire and forget operation. In the case of a conflict due to code that is merged before a given PR, you will need to troubleshoot your code manually.
# Manually Running Tests
Tests can also be run manually on the command line within a docker container, which can be built with:
   ```
   docker build -t tflm-ci -f ci/Dockerfile.micro .
   ```

   or use the tflm-ci docker image from [here](https://github.com/users/TFLM-bot/packages/container/package/tflm-ci).

  You will still need to copy or mount your fork of tflite-micro on to this docker container prior to running any tests. To run the built Docker image interactively and mount your local copy of tflite-micro on the container, run:
  ```
  docker run -v /path/to/local/tflite-micro:/path/to/docker/tflite-micro -it tflm-ci /bin/bash
  ```
  This way changes from your local fork will be reflected in the Docker container.

  You can also view or remove your instantiated containers by doing:
  ```
  docker ps --all
  docker rm <docker image ID>
  ```
# Sync From The Tensorflow Repository
While TfLite Micro and TfLite are in separate GitHub repositories, the two
projects continue to share common code.

The [TensorFlow repo](https://github.com/tensorflow/tensorflow) is the single source of truth for this
shared code. As a result, any changes to this shared code must be made in the
[TensorFlow repo](https://github.com/tensorflow/tensorflow) which will then automatically sync'd via a scheduled
[GitHub workflow](../.github/workflows/sync.yml).
# Merge Queue Details
This section is probably only of interest if you plan to be doing surgery on the CI system.
## Mergify
We use [Mergify](https://mergify.com/) for our merge queue. [The documentation](https://docs.mergify.com/) is reasonably straight forward.
## Config File
Our [mergify.yml](../.github/mergify.yml) is fairly standard. It triggers on `ci:ready_to_merge` label and requires all branch protection checks to pass before a PR can be added to the queue. When the PR is merged it removes the label.
## `ci:run` In Queue
The one slightly complicated wrinkle in our system is the test suite being run only when the `ci:run` label is applied. As soon as the tests are run, the label is [removed](../.github/workflows/remove-labels.yml).

As the queue processes each PR, it creates a temporary branch and merges in the results of the previous passing PRs my merging main. This merge resets all the test results, so the `ci:run` tag must be [reapplied](../.github/workflows/apply_cirun.yml) in queue.
