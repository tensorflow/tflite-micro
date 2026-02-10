# Continuous Integration

This document outlines the Continuous Integration (CI) system for TFLite Micro, including how to trigger tests, the merge queue process, and how to run tests locally.

## CI Workflow

The CI system behaves differently depending on your role. This "Tiered" approach ensures security for external contributions while maintaining velocity for maintainers.

### 1. Maintainers & Trusted Contributors
*   **Automatic Execution**: All tests (Basic and Privileged) run automatically on every push.
*   **No Approval Needed**: The system recognizes your permissions and bypasses the environment gate.

### 2. External Contributors
*   **Basic Tests**: Low-risk checks (Linting, Windows builds, File checks) run automatically on every push.
*   **Privileged Tests**: Hardware-in-the-loop tests (Cortex-M, Xtensa, Hexagon, RISC-V) require manual approval.
    *   **Status**: Your PR will show a "Pending" status for the `approval-gate` job.
    *   **Action**: A maintainer must review the code and click **"Review Deployment"** -> **"Approve"** on the PR page.
    *   **New Pushes**: Pushing new code resets the approval, requiring a maintainer to re-approve the specific commit.

## Labels

| Label | Description |
| :--- | :--- |
| `ci:ready` | **Request Review**: For external contributors, adding this label signals to maintainers that the PR is ready for review. It acts as a "noise filter" so maintainers aren't notified about draft PRs. |
| `ci:full` | **Extended Scope**: By default, CI runs a `basic` set of tests. Adding this label expands the scope to include all hardware targets (e.g., RISC-V, Hexagon). This scope change applies to both maintainers and external contributors. |

## GitHub Merge Queue

We use [GitHub Merge Queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue) to manage landings.

*   **Process**: Once a PR is approved and all required checks pass, maintainers click **"Merge when ready"** to add it to the queue.
*   **Validation**: The [merge_group.yml](../.github/workflows/merge_group.yml) workflow runs the full test suite on a temporary merge branch. This ensures that the combination of multiple PRs doesn't break `main` before they are merged.

## Sync From TensorFlow

TFLite Micro shares code with the main TensorFlow repository.

*   **Source of Truth**: The [TensorFlow repo](https://github.com/tensorflow/tensorflow) is the source for shared code.
*   **Process**: Any changes to shared code **must** be made in the TensorFlow repo first. They are automatically synced to TFLite Micro via the scheduled [sync.yml](../.github/workflows/sync.yml) workflow.

## Manually Running Tests (Docker)

You can reproduce CI environments locally using the TFLM CI Docker container.

1.  **Build the image**:
    ```bash
    docker build -t tflm-ci -f ci/Dockerfile.micro .
    ```
    *Alternatively, pull the pre-built image from [GitHub Packages](https://github.com/users/TFLM-bot/packages/container/package/tflm-ci).*

2.  **Run interactively**:
    Mount your local TFLite Micro directory to the container to run tests against your changes:
    ```bash
    docker run -v /path/to/local/tflite-micro:/path/to/docker/tflite-micro -it tflm-ci /bin/bash
    ```

3.  **Cleanup**:
    View and remove stopped containers:
    ```bash
    docker ps --all
    docker rm <docker image ID>
    ```
