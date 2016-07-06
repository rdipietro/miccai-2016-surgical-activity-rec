#!/bin/bash

data_dir="~/Data/MISTIC"

# 15 total users. 03 04 10 30 for validation. The rest
# for the test set.

for model_type in BidirectionalLSTM ForwardLSTM
do
    # Each test-users set consists of a single user.
    for test_users in 02 05 07 09 11 12 13 15 17 24 31
    do
        sbatch --time=0-16:0:0 run_one_job.sh \
            --data_dir "$data_dir" \
            --model_type "$model_type" \
            --test_users "$test_users"
    done
done
