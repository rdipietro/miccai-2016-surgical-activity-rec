#!/bin/bash

data_dir="~/Data/JIGSAWS/Suturing"

for model_type in BidirectionalLSTM ForwardLSTM
do
    # Each test-users set consists of a single user.
    for test_users in B C D E F G H I
    do
        bash run_one_job.sh \
            --data_dir "$data_dir" \
            --model_type "$model_type" \
            --test_users "$test_users"
    done
done
