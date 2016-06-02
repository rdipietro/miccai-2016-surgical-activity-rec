#!/bin/bash

export data_dir="~/Data/MISTIC"

# 15 total users. 03 04 10 30 for validation. The rest
# for the test set.

for model_type in BidirectionalLSTM ForwardLSTM
do
    export model_type
    # Each test-users set consists of a single user.
    for test_users in 02 05 07 09 11 12 13 15 17 24 31
    do
        export test_users
        sbatch run_one_train_eval.sh "mistic $model_type $test_users"
    done
done

