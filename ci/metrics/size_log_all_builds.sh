#!/bin/bash
# Traverse through Git log, check out and profile each commit
# Traverse backwards in time since the most recent commits are likely more interesting
LOCAL_TOP_FOLDER=~/local/
cd ../../../cmsis
LIST_OF_COMMITS=$(git rev-list upstream/develop)
TMP_ARRAY=($LIST_OF_COMMITS)
COUNTER=${#TMP_ARRAY[@]}
for COMMIT in $LIST_OF_COMMITS
do
    echo $COUNTER
    FOLDER_NAME=$(printf %03d $COUNTER)
    git checkout -f $COMMIT
    ../../core_software/tflite_micro/ci/metrics/create_size_log_CMSIS_NN.sh ${LOCAL_TOP_FOLDER}${FOLDER_NAME}
    COUNTER=$((COUNTER-1))
done
