* [Quick Start](#quick-start)
* [ci:run](#ci:run)
* [mergify.yml](#mergify.yml)
* [ci:run application in queue](ci:run-application-in-queue)
* [Viewing a PRs Merge Status](viewing-a-prs-merge-queue-status)
# Continuious Integration and Merge Queues
The tflite-micro github repo uses a number of Github Actions workflows to run tests as well as [Mergify](https://mergify.com/) to provide merge queue functionality. This document explains the setup and interactions between the two. 

# Quick Start
While there are a few moving parts to the Continuous Integration (CI) test runners and queuing system, usage is only two steps:
  
  * Apply the `ci:run` label to the PR
  * Apply the `ci:ready_to_merge` label to the PR

The order of application of these labels doesn't matter, but `ci:ready_to_merge` will have no practical effect until `ci:run` is applied and the test suite passes.

# ci:run
Applied to a PR, the `ci:run` label launches a number of Github Actions workflows located in .github/workflows. You can grep for '`ci:run`' in that directory to trace through the gory details. At the time of this writing five workflows are run, but that is a moving target as the test system is under active development.

The tests run by the `ci:run` trigger are required by the repository Branch Protection Rules for the PR to be merged into main.

As soon as it launches the tests, but before they finish, the `ci:run` label is removed from the PR. This is a slight convenience in situations where a PR might have multiple test runs due to requested changes before it is approved for merging.

# mergify.yml
Mergify is a third party merge queue system. The configuration file is located at .github/mergify.yml

You can find the full documentation on the Mergify system [here](https://docs.mergify.com/). The important parts of mergify.yml for our purposes are the conditional blocks under `queue_rules` and `pull_request_rules`. 

When the conditions under `pull_request_rules` are met the PR will be placed in the merge queue. It will be removed from the queue if the conditions under `queue_rules` fail. 

The queue goes in order and when the current PR at the head of the queue has met all the requirements of the repository Branch Protection Rules it will be merged with with main. In our case that means one approved review and all of the CI tests triggered by `ci:run` pass.

# ci:run Application in Queue
When a PR is one of a number in the queue, when previous PRs are merged into main the queue machinery to merge main into our PR, causing the test status reset. Since our tests are only run when the `ci:run` label is applied, the queue will stall waiting for the ci tests to pass.

The action workflow .github/workflows/apply_cirun.yml reapplies the `ci:run` label on merge events if a PR has the `ci:ready_to_merge` label to rerun the ci test suite and allow the queue to continue.

# Viewing a PRs Merge Queue Status
In the status area of an open PR, there will be a line for Mergify. If the PR is in queue, it will show the position in the queue. You can click on this line to be taken to the mergify status page for the particular PR, which will provide information including which checks have passed and which are still in process.